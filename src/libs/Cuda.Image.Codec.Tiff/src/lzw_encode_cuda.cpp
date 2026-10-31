/*
 * File: lzw_encode_cuda.cpp
 * Brief: Lzw encode cuda.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_encode_cuda.cpp - TIFF LZW CUDA 编码器实现。
#include "lzw_encode_cuda.h"

#include <algorithm>
#include <cstring>

#include "cuda_error.h"
#include "lzw_encode_launch.h"

namespace image_codec {
namespace cuda {
namespace tiff {

namespace {

// 单次 kernel 的显存预算：过大没有收益，过小会让大图分很多批反而更慢。
// 兜底：既没有预算也没有设备信息时，单批最多吃 1.5 GiB 显存。
// 真正的上限由 MemoryBudget / 设备空闲显存决定（见 batchBytesBudget）。
constexpr size_t kFallbackBatchBytes = static_cast<size_t>(1536) << 20;

} // namespace

LzwEncoderCuda::LzwEncoderCuda(int device_index) : device_index_(device_index) {
  bindBudget(nullptr);
}

void LzwEncoderCuda::setBudget(MemoryBudget *budget) { bindBudget(budget); }

// 预算由上层（C API 句柄）注入；未注入时用对象内部的记账器。
void LzwEncoderCuda::bindBudget(MemoryBudget *budget) {
  budget_ = budget != nullptr ? budget : &own_budget_;
  in_.setBudget(budget_);
  dict_.setBudget(budget_);
  out_.setBudget(budget_);
  sizes_.setBudget(budget_);
}

// 单批可用的显存：优先用预算上限，其次用设备真实空闲（留 25% 给别的分配），
// 最后才退到兜底常量。取一半是给 dict_/out_/sizes_ 以及驱动留余量。
size_t LzwEncoderCuda::batchBytesBudget() const {
  size_t limit = budget_ != nullptr ? budget_->limit() : 0;
  if (limit == 0) {
    DeviceMemoryInfo info;
    if (queryDeviceMemory(device_index_, &info, nullptr) &&
        info.free_bytes > 0) {
      limit = info.free_bytes / 4 * 3; // 留 25%
    }
  }
  if (limit == 0) {
    return kFallbackBatchBytes;
  }
  return std::max<size_t>(limit / 2, static_cast<size_t>(64) << 20);
}

LzwEncoderCuda::~LzwEncoderCuda() = default;

bool LzwEncoderCuda::initOnce() {
  if (init_done_) return true;
  if (!dev_.init(device_index_, &err_)) {
    return false;
  }
  init_done_ = true;
  return true;
}

int LzwEncoderCuda::encodeStrips(const ImageBuffer &img, int rows_per_strip,
                                 std::vector<uint32_t> &out_sizes,
                                 std::vector<uint8_t> &out_data) {
  out_sizes.clear();
  out_data.clear();
  if (!img.valid() || rows_per_strip <= 0) {
    err_ = "encodeStrips: invalid image or rows_per_strip";
    return -2;
  }
  if (rows_per_strip > img.height) rows_per_strip = img.height;
  const size_t row_bytes = img.stride();
  const size_t unit_bytes = static_cast<size_t>(rows_per_strip) * row_bytes;
  const size_t n_units = static_cast<size_t>(img.height) / rows_per_strip;

  // 整 strip 部分是等长单元，按批上传编码。
  if (n_units > 0) {
    const int rc = encodePaddedUnits(img.data.data(), unit_bytes, n_units,
                                     out_sizes, out_data);
    if (rc != 0) return rc;
  }

  // 尾部不足一个 strip：TIFF 规范允许最后一个 strip 行数较少，因此单独编一个
  // 单元，而不是像以前那样整幅图退回 CPU（大图那一退就是几十秒）。
  const int tail_rows =
      img.height -
      static_cast<int>(n_units * static_cast<size_t>(rows_per_strip));
  if (tail_rows > 0) {
    const uint8_t *tail =
        img.data.data() +
        n_units * static_cast<size_t>(rows_per_strip) * row_bytes;
    std::vector<uint32_t> tail_sizes;
    std::vector<uint8_t> tail_data;
    const int rc =
        encodePaddedUnits(tail, static_cast<size_t>(tail_rows) * row_bytes, 1,
                          tail_sizes, tail_data);
    if (rc != 0) return rc;
    out_sizes.insert(out_sizes.end(), tail_sizes.begin(), tail_sizes.end());
    out_data.insert(out_data.end(), tail_data.begin(), tail_data.end());
  }
  return 0;
}

int LzwEncoderCuda::encodeTiles(const ImageBuffer &img, int tile_size,
                                std::vector<uint32_t> &out_sizes,
                                std::vector<uint8_t> &out_data) {
  out_sizes.clear();
  out_data.clear();
  if (!img.valid() || tile_size <= 0 || (tile_size & 15) != 0) {
    err_ = "encodeTiles: invalid image or tile_size not a multiple of 16";
    return -2;
  }
  const int tw = tile_size;
  const int th = tile_size;
  const int tiles_x = (img.width + tw - 1) / tw;
  const int tiles_y = (img.height + th - 1) / th;
  const size_t n_units =
      static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y);
  const size_t tile_row_bytes = static_cast<size_t>(tw) * img.channels;
  const size_t unit_bytes = tile_row_bytes * th;

  // tile 需要把边缘补齐到整块（TIFF 规范要求 tile 数据为完整块大小）。
  // 分批构造 -> 上传 -> kernel -> 回收，避免整幅 padded 副本常驻内存。
  std::vector<uint8_t> host_tile(unit_bytes, 0);
  const size_t per_unit = unit_bytes + 4096ull * 4 + (unit_bytes * 3 / 2 + 64);
  size_t cap = unit_batch_ > 0
                   ? static_cast<size_t>(unit_batch_)
                   : batchBytesBudget() / std::max<size_t>(per_unit, 1);
  cap = std::max<size_t>(cap, 1);
  cap = std::min<size_t>(cap, static_cast<size_t>(4096));

  std::vector<uint32_t> chunk_sizes;
  std::vector<uint8_t> chunk_out;
  for (size_t base = 0; base < n_units; base += cap) {
    const size_t count = std::min(cap, n_units - base);
    std::vector<uint8_t> batch(count * unit_bytes);
    for (size_t i = 0; i < count; ++i) {
      const size_t idx = base + i;
      const int tx = static_cast<int>(idx % static_cast<size_t>(tiles_x));
      const int ty = static_cast<int>(idx / static_cast<size_t>(tiles_x));
      const int x0 = tx * tw;
      const int y0 = ty * th;
      const int w_valid = std::min(tw, img.width - x0);
      const int h_valid = std::min(th, img.height - y0);
      std::fill(host_tile.begin(), host_tile.end(), 0);
      for (int r = 0; r < h_valid; ++r) {
        const uint8_t *src =
            img.data.data() + (static_cast<size_t>(y0 + r) * img.stride() +
                               static_cast<size_t>(x0) * img.channels);
        std::memcpy(host_tile.data() + static_cast<size_t>(r) * tile_row_bytes,
                    src, static_cast<size_t>(w_valid) * img.channels);
      }
      std::memcpy(batch.data() + i * unit_bytes, host_tile.data(), unit_bytes);
    }
    const int rc = encodePaddedUnits(batch.data(), unit_bytes, count,
                                     chunk_sizes, chunk_out);
    if (rc != 0) {
      out_sizes.clear();
      out_data.clear();
      return rc;
    }
    out_sizes.insert(out_sizes.end(), chunk_sizes.begin(), chunk_sizes.end());
    out_data.insert(out_data.end(), chunk_out.begin(), chunk_out.end());
  }
  return 0;
}

int LzwEncoderCuda::encodePaddedUnits(const uint8_t *input, size_t unit_bytes,
                                      size_t n_units,
                                      std::vector<uint32_t> &out_sizes,
                                      std::vector<uint8_t> &out_data) {
  if (!initOnce()) return -1;
  out_sizes.clear();
  out_data.clear();
  if (input == nullptr || unit_bytes == 0 || n_units == 0) {
    err_ = "encodeUnits: empty input";
    return -2;
  }
  const size_t max_unit = unit_bytes * 3 / 2 + 64;
  if (max_unit > 0xFFFFFFFFull) {
    err_ = "encodeUnits: unit too large";
    return -2;
  }
  const size_t per_unit = unit_bytes + 4096ull * 4 + max_unit;
  size_t cap = unit_batch_ > 0
                   ? static_cast<size_t>(unit_batch_)
                   : batchBytesBudget() / std::max<size_t>(per_unit, 1);
  cap = std::max<size_t>(cap, 1);
  cap = std::min<size_t>(cap, static_cast<size_t>(4096));

  std::vector<uint32_t> sizes_host;
  std::vector<uint8_t> out_host;
  for (size_t base = 0; base < n_units; base += cap) {
    const size_t count = std::min(cap, n_units - base);
    last_batch_units_ = count;
    if (!in_.upload(input + base * unit_bytes, count * unit_bytes, &err_,
                    dev_.stream())) {
      return -3;
    }
    if (!dict_.ensureItems<uint32_t>(count * 4096u, &err_) ||
        !out_.ensure(count * max_unit, &err_) ||
        !sizes_.ensureItems<uint32_t>(count, &err_)) {
      return -3;
    }
    launchLzwEncode(in_.as<uint8_t>(), dict_.as<uint32_t>(), out_.as<uint8_t>(),
                    sizes_.as<uint32_t>(), static_cast<uint32_t>(unit_bytes),
                    static_cast<uint32_t>(max_unit),
                    static_cast<uint32_t>(count), block_threads_,
                    dev_.stream());
    if (!check(cudaGetLastError(), "lzw_encode_kernel", &err_)) {
      return -4;
    }
    if (!dev_.synchronize(&err_)) {
      return -4;
    }
    sizes_host.resize(count);
    if (!sizes_.download(sizes_host.data(), count * sizeof(uint32_t), &err_,
                         dev_.stream())) {
      return -5;
    }
    size_t chunk_total = 0;
    for (size_t i = 0; i < count; ++i) {
      if (sizes_host[i] == 0 || sizes_host[i] > max_unit) {
        err_ = "lzw_encode_kernel: incoherent unit size";
        return -5;
      }
      chunk_total += sizes_host[i];
    }
    out_host.resize(chunk_total);
    // 逐单元回收：unit 之间在 device 上是定长 stride，不能整体一次拷贝。
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!check(cudaMemcpyAsync(out_host.data() + offset,
                                 out_.as<uint8_t>() + i * max_unit,
                                 sizes_host[i], cudaMemcpyDeviceToHost,
                                 dev_.stream()),
                 "cudaMemcpyAsync D2H", &err_)) {
        return -6;
      }
      offset += sizes_host[i];
    }
    if (!dev_.synchronize(&err_)) {
      return -6;
    }
    out_sizes.insert(out_sizes.end(), sizes_host.begin(), sizes_host.end());
    out_data.insert(out_data.end(), out_host.begin(), out_host.end());
  }
  return 0;
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
