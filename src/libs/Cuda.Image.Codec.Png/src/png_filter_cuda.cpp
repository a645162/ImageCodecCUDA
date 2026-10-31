/*
 * File: png_filter_cuda.cpp
 * Brief: Png filter cuda.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_filter_cuda.cpp - PNG 行滤波 CUDA 实现。
#include "png_filter_cuda.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "cuda_error.h"
#include "png_filter_launch.h"

namespace image_codec {
namespace cuda {
namespace png {

PngFilterCuda::PngFilterCuda(int device_index) : device_index_(device_index) {
  bindBudget(nullptr);
}

void PngFilterCuda::setBudget(MemoryBudget *budget) { bindBudget(budget); }

// 预算由上层（C API 句柄）注入；未注入时用对象内部的记账器。
void PngFilterCuda::bindBudget(MemoryBudget *budget) {
  budget_ = budget != nullptr ? budget : &own_budget_;
  src_.setBudget(budget_);
  dst_.setBudget(budget_);
  types_.setBudget(budget_);
  sums_.setBudget(budget_);
}

PngFilterCuda::~PngFilterCuda() = default;

bool PngFilterCuda::initOnce() {
  if (init_done_) return true;
  if (!dev_.init(device_index_, &err_)) {
    return false;
  }
  init_done_ = true;
  return true;
}

bool PngFilterCuda::buildFilteredStream(const ImageBuffer &img, int filter_type,
                                        std::vector<uint8_t> *out,
                                        std::vector<uint8_t> *filters_out) {
  if (out == nullptr) {
    err_ = "PNG filter: null output";
    return false;
  }
  out->clear();
  if (filters_out != nullptr) filters_out->clear();
  if (!img.valid()) {
    err_ = "PNG filter: invalid image buffer";
    return false;
  }
  if (filter_type < -1 || filter_type > 4) {
    err_ = "PNG filter: filter_type out of range";
    return false;
  }
  if (!initOnce()) {
    return false;
  }

  const size_t row_bytes = img.stride();
  const size_t filtered_row = row_bytes + 1;
  const size_t total = filtered_row * static_cast<size_t>(img.height);

  std::vector<uint8_t> filters;
  if (filter_type < 0) {
    // 1) GPU 统计 5 种滤波器的残差绝对值和。
    if (!sums_.ensureItems<uint64_t>(static_cast<size_t>(img.height) * 5,
                                     &err_)) {
      return false;
    }
    if (!src_.upload(img.data.data(), img.data.size(), &err_, dev_.stream())) {
      return false;
    }
    launchPngFilterSums(src_.as<uint8_t>(), sums_.as<uint64_t>(), img.width,
                        img.height, img.channels, block_threads_,
                        dev_.stream());
    if (!check(cudaGetLastError(), "png_filter_sums_kernel", &err_) ||
        !dev_.synchronize(&err_)) {
      return false;
    }
    std::vector<uint64_t> host_sums(static_cast<size_t>(img.height) * 5);
    if (!sums_.download(host_sums.data(), host_sums.size() * sizeof(uint64_t),
                        &err_, dev_.stream())) {
      return false;
    }
    // 2) 逐行选最小（height 行，纯标量，代价可忽略）。
    filters.resize(static_cast<size_t>(img.height));
    for (int y = 0; y < img.height; ++y) {
      uint64_t best = std::numeric_limits<uint64_t>::max();
      int best_f = 0;
      for (int f = 0; f < 5; ++f) {
        const uint64_t v = host_sums[static_cast<size_t>(y) * 5 + f];
        if (v < best) {
          best = v;
          best_f = f;
        }
      }
      filters[static_cast<size_t>(y)] = static_cast<uint8_t>(best_f);
    }
    if (!types_.upload(filters.data(), filters.size(), &err_, dev_.stream())) {
      return false;
    }
  }

  if (!dst_.ensure(total, &err_)) {
    return false;
  }
  if (filter_type >= 0) {
    if (!src_.upload(img.data.data(), img.data.size(), &err_, dev_.stream())) {
      return false;
    }
  }
  launchPngFilter(src_.as<uint8_t>(), dst_.as<uint8_t>(),
                  filter_type < 0 ? types_.as<uint8_t>() : nullptr, img.width,
                  img.height, img.channels, filter_type, block_threads_,
                  dev_.stream());
  if (!check(cudaGetLastError(), "png_filter_rows_kernel", &err_) ||
      !dev_.synchronize(&err_)) {
    return false;
  }
  out->resize(total);
  if (!dst_.download(out->data(), total, &err_, dev_.stream())) {
    out->clear();
    return false;
  }
  if (filters_out != nullptr) {
    *filters_out = std::move(filters);
  }
  return true;
}

} // namespace png
} // namespace cuda
} // namespace image_codec
