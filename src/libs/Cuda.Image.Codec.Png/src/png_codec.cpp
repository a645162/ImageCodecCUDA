/*
 * File: png_codec.cpp
 * Brief: Png codec.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_codec.cpp - PNG 编解码门面实现。
#include "png_codec.h"

#include <memory>

#include "png_container.h"
#include "png_filter_cuda.h"

namespace image_codec {
namespace cuda {
namespace png {

bool encodePng(const ImageBuffer &img, const PngEncodeOptions &options,
               std::vector<uint8_t> *out, bool *gpu_used, std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "PNG: null output";
    return false;
  }
  out->clear();
  if (gpu_used != nullptr) *gpu_used = false;
  if (!img.valid()) {
    if (err != nullptr) *err = "PNG: invalid image buffer";
    return false;
  }

  std::vector<uint8_t> filtered;
  if (options.use_gpu) {
    PngFilterCuda filter;
    filter.setBudget(options.budget);
    filter.setBlockThreads(options.block_threads);
    if (filter.buildFilteredStream(img, options.filter, &filtered, nullptr)) {
      if (gpu_used != nullptr) *gpu_used = true;
    } else {
      // 明确记录原因；显存不足时默认回退 CPU 参考实现（仍产出合法 PNG），
      // 但绝不静默——err 里必须能看出为什么没走 GPU。
      if (err != nullptr)
        *err = "PNG: CUDA filtering failed: " + filter.lastError();
      filtered.clear();
      if (!options.allow_cpu_fallback) {
        return false;
      }
    }
  }
  if (filtered.empty()) {
    std::vector<uint8_t> filters;
    if (options.filter >= 0) {
      filters.assign(static_cast<size_t>(img.height),
                     static_cast<uint8_t>(options.filter));
    } else {
      filters = chooseAdaptiveFilters(img.data.data(), img.width, img.height,
                                      img.channels);
    }
    filtered = buildFilteredStream(img.data.data(), img.width, img.height,
                                   img.channels, filters);
  }
  if (filtered.empty()) {
    if (err != nullptr && err->empty()) *err = "PNG: filtering failed";
    return false;
  }

  std::string write_err;
  if (!writeFilteredPng(filtered.data(), filtered.size(), img.width, img.height,
                        img.channels, options.compression_level, *out,
                        &write_err)) {
    if (err != nullptr) *err = "PNG: " + write_err;
    return false;
  }
  return true;
}

bool decodePng(const uint8_t *data, size_t size, ImageBuffer *out,
               std::string *err) {
  return readPng(data, size, *out, err);
}

} // namespace png
} // namespace cuda
} // namespace image_codec
