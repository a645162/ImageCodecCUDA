/*
 * File: png_codec.h
 * Brief: Png codec.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_codec.h - PNG 编解码门面（编码滤波在 GPU，deflate/inflate 在 zlib）。
//
// 解码为什么在 CPU：unfilter 的依赖链是「同一行左侧 → 上一行」，每一行都依赖
// 前一行的最终结果，并行度只有行内一度；把这条链搬到 GPU 只会多两次 H2D/D2H。
// zlib inflate 同样如此。GPU 在这里的价值是编码期的逐行滤波选择与滤波计算。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda_memory.h"
#include "image_buffer.h"
#include "png_container.h"

namespace image_codec {
namespace cuda {
namespace png {

struct PngEncodeOptions {
  int compression_level = 6; // zlib 0..9
  int filter = -1;           // -1 自适应，0..4 固定
  bool use_gpu = true;       // false 时用 Common 里的 CPU 参考滤波
  // 调参旋钮：一行一个 block 的线程数（0 = 256）。不同 GPU 上最优值不同。
  int block_threads = 0;
  // 显存预算（nullptr = 不设上限，仅设备空闲显存约束）。
  MemoryBudget *budget = nullptr;
  // 显存不足时是否允许回退到 CPU 滤波（默认 true：产出合法 PNG 优先）。
  bool allow_cpu_fallback = true;
};

// 编码。gpu_used（可为 nullptr）回报本次滤波是否真的在 GPU 上完成。
bool encodePng(const ImageBuffer &img, const PngEncodeOptions &options,
               std::vector<uint8_t> *out, bool *gpu_used, std::string *err);

// 解码（CPU）。
bool decodePng(const uint8_t *data, size_t size, ImageBuffer *out,
               std::string *err);

} // namespace png
} // namespace cuda
} // namespace image_codec
