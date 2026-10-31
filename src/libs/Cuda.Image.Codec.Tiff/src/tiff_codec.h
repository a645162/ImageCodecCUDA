// tiff_codec.h - TIFF 编解码门面（容器 + LZW；编码可走 GPU，解码走 CPU）。
//
// 解码为什么在 CPU：LZW 解码是严格串行依赖（字典条目依赖前一条输出），
// 在 8-bit chunky 图像上 GPU 解码始终慢于标量 CPU（ImageCodecOpenCL 的
// 早期探索结论一致），因此这里不再提供一条「看起来有 GPU 参与」的慢路径。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda_memory.h"
#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace tiff {

struct TiffEncodeOptions {
  int compression = 5;    // 1 = none, 5 = LZW
  int rows_per_strip = 0; // 0 = 自动（strip 布局时按 64 行）
  int tile_size = 0;      // >0 = tiled 布局（16 的倍数）；GPU LZW 推荐 256/512
  bool use_gpu = true;    // LZW 是否走 CUDA kernel

  // ---- 调参旋钮（Tune 库会扫这些值）----
  // 每个 block 的线程数（一个线程一个 strip/tile），0 = 默认 128。
  int lzw_block_threads = 0;
  // 每批上传的单元数，0 = 按显存预算自动。
  int lzw_unit_batch = 0;
  // 显存预算（nullptr = 不设上限，仅受设备空闲显存约束）。
  MemoryBudget *budget = nullptr;
  // GPU 失败（含显存不足）时是否回退 CPU LZW。false 时直接返回错误，
  // 让上层 router 自己决定换后端，而不是在这里悄悄慢下来。
  bool allow_cpu_fallback = false;
};

// 编码为 TIFF 字节流。gpu_used（可为 nullptr）回报本次是否真的用了 GPU。
bool encodeTiff(const ImageBuffer &img, const TiffEncodeOptions &options,
                std::vector<uint8_t> *out, bool *gpu_used, std::string *err);

// 解码 TIFF 字节流（8-bit chunky，compression none/LZW，灰度/RGB/RGBA）。
bool decodeTiff(const uint8_t *data, size_t size, ImageBuffer *out,
                std::string *err);

// 探测魔数（II*\0 / MM\0*）。
inline bool hasTiffSignature(const uint8_t *p, size_t n) {
  if (p == nullptr || n < 4) return false;
  const bool le = p[0] == 'I' && p[1] == 'I' && p[2] == 42 && p[3] == 0;
  const bool be = p[0] == 'M' && p[1] == 'M' && p[2] == 0 && p[3] == 42;
  return le || be;
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
