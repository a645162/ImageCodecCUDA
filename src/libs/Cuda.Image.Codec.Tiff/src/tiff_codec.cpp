/*
 * File: tiff_codec.cpp
 * Brief: Tiff codec.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tiff_codec.cpp - TIFF 编解码实现。
#include "tiff_codec.h"

#include <algorithm>
#include <cstring>

#include "lzw_codec.h"
#include "lzw_encode_cuda.h"
#include "tiff_container.h"

namespace image_codec {
namespace cuda {
namespace tiff {

namespace {

// 让 rows_per_strip 整除 height（GPU strip 并行要求等长）。
int clampRowsPerStrip(int height, int rows_per_strip) {
  if (rows_per_strip <= 0) {
    rows_per_strip = 64;
  }
  rows_per_strip = std::min(rows_per_strip, height);
  if (height % rows_per_strip == 0) {
    return rows_per_strip;
  }
  for (int r = rows_per_strip; r >= 1; --r) {
    if (height % r == 0) {
      return r;
    }
  }
  return 1;
}

// 把整幅图按 tile 切片并 padding 成完整 tile（无损，供写容器用）。
std::vector<uint8_t> buildPaddedTiles(const ImageBuffer &img, int tile_size,
                                      size_t *tile_bytes) {
  const int tw = tile_size;
  const int th = tile_size;
  const int tiles_x = (img.width + tw - 1) / tw;
  const int tiles_y = (img.height + th - 1) / th;
  const size_t tile_row = static_cast<size_t>(tw) * img.channels;
  const size_t unit = tile_row * th;
  *tile_bytes = unit;
  std::vector<uint8_t> tiles(
      unit * static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y), 0);
  for (int ty = 0; ty < tiles_y; ++ty) {
    for (int tx = 0; tx < tiles_x; ++tx) {
      const size_t idx = static_cast<size_t>(ty) * tiles_x + tx;
      uint8_t *dst = tiles.data() + idx * unit;
      const int x0 = tx * tw;
      const int y0 = ty * th;
      const int w_valid = std::min(tw, img.width - x0);
      const int h_valid = std::min(th, img.height - y0);
      for (int r = 0; r < h_valid; ++r) {
        const uint8_t *src =
            img.data.data() + (static_cast<size_t>(y0 + r) * img.stride() +
                               static_cast<size_t>(x0) * img.channels);
        std::memcpy(dst + static_cast<size_t>(r) * tile_row, src,
                    static_cast<size_t>(w_valid) * img.channels);
      }
    }
  }
  return tiles;
}

} // namespace

bool encodeTiff(const ImageBuffer &img, const TiffEncodeOptions &options,
                std::vector<uint8_t> *out, bool *gpu_used, std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "TIFF: null output";
    return false;
  }
  out->clear();
  if (gpu_used != nullptr) *gpu_used = false;
  if (!img.valid()) {
    if (err != nullptr) *err = "TIFF: invalid image buffer";
    return false;
  }
  if (options.compression != 1 && options.compression != 5) {
    if (err != nullptr) *err = "TIFF: unsupported compression";
    return false;
  }
  const uint16_t compression =
      options.compression == 5 ? kCompressionLzw : kCompressionNone;

  // ---- tiled 布局 ----
  if (options.tile_size > 0) {
    const int ts = options.tile_size;
    if (ts % 16 != 0) {
      if (err != nullptr) *err = "TIFF: tile_size must be a multiple of 16";
      return false;
    }
    const int tiles_x = (img.width + ts - 1) / ts;
    const int tiles_y = (img.height + ts - 1) / ts;
    const size_t n_tiles =
        static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y);

    std::vector<uint32_t> sizes;
    std::vector<uint8_t> data;
    bool gpu = false;
    if (options.compression == 1) {
      // 不压缩时每个 tile 都必须是完整块，只能构造整幅 padded 平面。
      size_t tile_bytes = 0;
      std::vector<uint8_t> tiles = buildPaddedTiles(img, ts, &tile_bytes);
      sizes.reserve(n_tiles);
      for (size_t i = 0; i < n_tiles; ++i) {
        sizes.push_back(static_cast<uint32_t>(tile_bytes));
      }
      data = std::move(tiles);
    } else if (options.use_gpu) {
      LzwEncoderCuda enc(0);
      enc.setBudget(options.budget);
      enc.setBlockThreads(options.lzw_block_threads);
      enc.setUnitBatch(options.lzw_unit_batch);
      // 流式：分批构造 padded tile -> 上传 -> kernel -> 回收。1.9 GB 的图
      // 若先构造整幅 padded 平面，会白白多占一份与图像等大的宿主内存。
      if (enc.encodeTiles(img, ts, sizes, data) != 0) {
        const std::string why =
            "TIFF: CUDA LZW encode failed: " + enc.lastError();
        if (!options.allow_cpu_fallback) {
          if (err != nullptr) *err = why;
          return false;
        }
        if (err != nullptr) *err = why + " (CPU fallback)";
        data.clear();
        sizes.clear();
        gpu = false;
      } else {
        gpu = true;
      }
    }
    if (!gpu && options.compression == 5 && data.empty()) {
      // 走到这里只可能是上面 GPU 路径失败并允许回退：用 CPU LZW 重编。
      size_t tile_bytes = 0;
      std::vector<uint8_t> tiles = buildPaddedTiles(img, ts, &tile_bytes);
      sizes.reserve(n_tiles);
      for (size_t i = 0; i < n_tiles; ++i) {
        std::vector<uint8_t> comp;
        if (lzw::encode(tiles.data() + i * tile_bytes, tile_bytes, comp) != 0) {
          if (err != nullptr) *err = "TIFF: CPU LZW encode failed";
          return false;
        }
        sizes.push_back(static_cast<uint32_t>(comp.size()));
        data.insert(data.end(), comp.begin(), comp.end());
      }
    }

    TiffWriter writer;
    if (!writer.openTile(img.width, img.height, img.channels,
                         static_cast<uint32_t>(ts), static_cast<uint32_t>(ts),
                         compression, n_tiles)) {
      if (err != nullptr) *err = "TIFF: writer openTile failed";
      return false;
    }
    size_t offset = 0;
    for (size_t i = 0; i < n_tiles; ++i) {
      if (!writer.appendTile(i, data.data() + offset, sizes[i])) {
        if (err != nullptr) *err = "TIFF: appendTile failed";
        return false;
      }
      offset += sizes[i];
    }
    *out = writer.buffer();
    if (gpu_used != nullptr) *gpu_used = gpu;
    return true;
  }

  // ---- strip 布局 ----
  const int rows_per_strip =
      clampRowsPerStrip(img.height, options.rows_per_strip);
  const size_t n_strips =
      (static_cast<size_t>(img.height) + rows_per_strip - 1) / rows_per_strip;

  std::vector<uint32_t> sizes;
  std::vector<uint8_t> data;
  bool gpu = false;
  if (options.compression == 1) {
    sizes.reserve(n_strips);
    for (size_t i = 0; i < n_strips; ++i) {
      const size_t y0 = i * static_cast<size_t>(rows_per_strip);
      const size_t rows =
          std::min<size_t>(static_cast<size_t>(rows_per_strip),
                           static_cast<size_t>(img.height) - y0);
      const size_t bytes = rows * img.stride();
      data.insert(data.end(), img.data.begin() + y0 * img.stride(),
                  img.data.begin() + y0 * img.stride() + bytes);
      sizes.push_back(static_cast<uint32_t>(bytes));
    }
  } else if (options.use_gpu) {
    LzwEncoderCuda enc(0);
    enc.setBudget(options.budget);
    enc.setBlockThreads(options.lzw_block_threads);
    enc.setUnitBatch(options.lzw_unit_batch);
    if (enc.encodeStrips(img, rows_per_strip, sizes, data) != 0) {
      const std::string why =
          "TIFF: CUDA LZW encode failed: " + enc.lastError();
      if (!options.allow_cpu_fallback) {
        if (err != nullptr) *err = why;
        return false;
      }
      if (err != nullptr) *err = why + " (CPU fallback)";
      sizes.clear();
      data.clear();
      gpu = false;
    } else {
      gpu = true;
    }
    if (!gpu) {
      sizes.reserve(n_strips);
      for (size_t i = 0; i < n_strips; ++i) {
        const size_t y0 = i * static_cast<size_t>(rows_per_strip);
        const size_t rows =
            std::min<size_t>(static_cast<size_t>(rows_per_strip),
                             static_cast<size_t>(img.height) - y0);
        const size_t bytes = rows * img.stride();
        std::vector<uint8_t> comp;
        if (lzw::encode(img.data.data() + y0 * img.stride(), bytes, comp) !=
            0) {
          if (err != nullptr) *err = "TIFF: CPU LZW encode failed";
          return false;
        }
        sizes.push_back(static_cast<uint32_t>(comp.size()));
        data.insert(data.end(), comp.begin(), comp.end());
      }
    }
  } else {
    sizes.reserve(n_strips);
    for (size_t i = 0; i < n_strips; ++i) {
      const size_t y0 = i * static_cast<size_t>(rows_per_strip);
      const size_t rows =
          std::min<size_t>(static_cast<size_t>(rows_per_strip),
                           static_cast<size_t>(img.height) - y0);
      const size_t bytes = rows * img.stride();
      std::vector<uint8_t> comp;
      if (lzw::encode(img.data.data() + y0 * img.stride(), bytes, comp) != 0) {
        if (err != nullptr) *err = "TIFF: CPU LZW encode failed";
        return false;
      }
      sizes.push_back(static_cast<uint32_t>(comp.size()));
      data.insert(data.end(), comp.begin(), comp.end());
    }
  }

  TiffWriter writer;
  if (!writer.openStrip(img.width, img.height, img.channels, rows_per_strip,
                        compression, n_strips)) {
    if (err != nullptr) *err = "TIFF: writer openStrip failed";
    return false;
  }
  size_t offset = 0;
  for (size_t i = 0; i < n_strips; ++i) {
    if (!writer.appendStrip(i, data.data() + offset, sizes[i])) {
      if (err != nullptr) *err = "TIFF: appendStrip failed";
      return false;
    }
    offset += sizes[i];
  }
  *out = writer.buffer();
  if (gpu_used != nullptr) *gpu_used = gpu;
  return true;
}

bool decodeTiff(const uint8_t *data, size_t size, ImageBuffer *out,
                std::string *err) {
  if (data == nullptr || size < 8 || out == nullptr) {
    if (err != nullptr) *err = "TIFF: empty input";
    return false;
  }
  TiffReader reader;
  if (!reader.openMemory(data, size)) {
    if (err != nullptr) *err = "TIFF: header/IFD parse failed";
    return false;
  }

  int w = 0;
  int h = 0;
  int spp = 1;
  int bps = 8;
  int photometric = 0;
  int compression = 1;
  int planar = 1;
  if (!reader.imageInfo(w, h, spp, bps, photometric, compression, planar)) {
    if (err != nullptr) *err = "TIFF: core tags missing";
    return false;
  }
  if (bps != 8 || planar != 1) {
    if (err != nullptr) *err = "TIFF: only 8-bit chunky is supported";
    return false;
  }
  if (photometric == 3) {
    if (err != nullptr) *err = "TIFF: palette images are not supported";
    return false;
  }
  const bool gray = photometric == 0 || photometric == 1;
  if (gray) {
    if (spp != 1 && spp != 2) {
      if (err != nullptr) *err = "TIFF: inconsistent samplesPerPixel for gray";
      return false;
    }
  } else if (photometric == 2) {
    if (spp != 3 && spp != 4) {
      if (err != nullptr) *err = "TIFF: inconsistent samplesPerPixel for RGB";
      return false;
    }
  } else {
    if (err != nullptr) *err = "TIFF: unsupported photometric";
    return false;
  }
  if (compression != kCompressionNone && compression != kCompressionLzw) {
    if (err != nullptr) *err = "TIFF: unsupported compression";
    return false;
  }
  if (!out->allocate(w, h, spp)) {
    if (err != nullptr) *err = "TIFF: allocation failed";
    return false;
  }
  const size_t row_bytes = out->stride();
  const size_t pixel_bytes = static_cast<size_t>(spp);

  auto decodeOne = [&](const uint8_t *raw, size_t raw_size,
                       std::vector<uint8_t> &dec) -> bool {
    if (compression == kCompressionLzw) {
      return lzw::decode(raw, raw_size, dec) == 0;
    }
    dec.assign(raw, raw + raw_size);
    return true;
  };

  if (reader.isTiled()) {
    std::vector<uint32_t> offs;
    std::vector<uint32_t> counts;
    uint32_t tw = 0;
    uint32_t th = 0;
    if (!reader.getTileInfo(offs, counts, tw, th)) {
      if (err != nullptr) *err = "TIFF: tile tags invalid";
      return false;
    }
    if (tw == 0 || th == 0) {
      if (err != nullptr) *err = "TIFF: zero tile size";
      return false;
    }
    const size_t tiles_per_row = (static_cast<size_t>(w) + tw - 1) / tw;
    const size_t tile_row_bytes = static_cast<size_t>(tw) * pixel_bytes;
    const size_t tile_bytes = tile_row_bytes * th;
    for (size_t i = 0; i < offs.size(); ++i) {
      std::vector<uint8_t> raw;
      if (!reader.loadBytes(offs[i], counts[i], raw)) {
        if (err != nullptr) *err = "TIFF: tile payload out of range";
        return false;
      }
      std::vector<uint8_t> dec;
      if (!decodeOne(raw.data(), raw.size(), dec)) {
        if (err != nullptr) *err = "TIFF: tile LZW decode failed";
        return false;
      }
      if (dec.size() < tile_bytes) {
        if (err != nullptr) *err = "TIFF: tile payload too small";
        return false;
      }
      const size_t tc = i % tiles_per_row;
      const size_t tr = i / tiles_per_row;
      const size_t x0 = tc * tw;
      const size_t y0 = tr * th;
      if (y0 >= static_cast<size_t>(h) || x0 >= static_cast<size_t>(w)) {
        continue;
      }
      const size_t w_valid =
          std::min(static_cast<size_t>(tw), static_cast<size_t>(w) - x0);
      const size_t h_valid =
          std::min(static_cast<size_t>(th), static_cast<size_t>(h) - y0);
      for (size_t r = 0; r < h_valid; ++r) {
        std::memcpy(out->data.data() + (y0 + r) * row_bytes + x0 * pixel_bytes,
                    dec.data() + r * tile_row_bytes, w_valid * pixel_bytes);
      }
    }
    return true;
  }

  std::vector<uint32_t> offs;
  std::vector<uint32_t> counts;
  uint32_t rps = 0;
  if (!reader.getStripInfo(offs, counts, rps)) {
    if (err != nullptr) *err = "TIFF: strip tags invalid";
    return false;
  }
  if (rps == 0) rps = 1;
  for (size_t i = 0; i < offs.size(); ++i) {
    std::vector<uint8_t> raw;
    if (!reader.loadBytes(offs[i], counts[i], raw)) {
      if (err != nullptr) *err = "TIFF: strip payload out of range";
      return false;
    }
    std::vector<uint8_t> dec;
    if (!decodeOne(raw.data(), raw.size(), dec)) {
      if (err != nullptr) *err = "TIFF: strip LZW decode failed";
      return false;
    }
    const size_t y0 = i * static_cast<size_t>(rps);
    if (y0 >= static_cast<size_t>(h)) {
      break;
    }
    const size_t rows =
        std::min<size_t>(static_cast<size_t>(rps), static_cast<size_t>(h) - y0);
    const size_t need = rows * row_bytes;
    const size_t copy_n = std::min(need, dec.size());
    std::memcpy(out->data.data() + y0 * row_bytes, dec.data(), copy_n);
  }
  return true;
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
