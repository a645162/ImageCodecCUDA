// jpeg_codec.h - baseline JPEG 编解码（nvJPEG）。
//
// 为什么用 nvJPEG 而不是自己写 DCT kernel：nvJPEG 在 Ada/Hopper 上使用专用
// NVJPG 硬件单元（解码）与优化过的熵编码路径，自己写的 compute kernel 在这
// 两件事上都不可能更快；本项目的价值是把 nvJPEG 包成与其它后端一致的语义
// （RGBI 内存进出、统一错误上下文、可选子采样），而不是重复实现它。
//
// 本模块不可用时（无 nvJPEG 组件的工具链）所有入口返回 false + 明确原因，
// 由上层回退到其它后端，绝不静默换实现。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda_memory.h"
#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace jpeg {

// 本编译单元是否链接了 nvJPEG。
bool available();

// 解码为 RGB（3 通道）。灰度 JPEG 也按 RGB 复制返回（与 ImageViewer 的
// 3 通道内存约定一致，且不引入有损的颜色空间转换）。
// budget 非空时，解码所需的设备缓冲（pitch 对齐的整幅图）先记账，超限即失败。
bool decode(int device_index, const uint8_t *data, size_t size,
            ImageBuffer *out, std::string *err, MemoryBudget *budget = nullptr);

// 编码。quality 1..100；subsampling: 0=按质量自动（>=92 → 444，否则 420），
// 否则 444 / 422 / 420。channels 支持 3（RGB）与 1（灰度，转 RGB 编码）。
bool encode(int device_index, const ImageBuffer &img, int quality,
            int subsampling, std::vector<uint8_t> *out, std::string *err,
            MemoryBudget *budget = nullptr);

// 魔数：SOI (FF D8) + 下一个标记 (FF)。
inline bool hasJpegSignature(const uint8_t *p, size_t n) {
  return p != nullptr && n >= 3 && p[0] == 0xFF && p[1] == 0xD8 && p[2] == 0xFF;
}

} // namespace jpeg
} // namespace cuda
} // namespace image_codec
