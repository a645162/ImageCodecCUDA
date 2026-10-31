/*
 * File: png_container.h
 * Brief: Png container.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_container.h - PNG 容器（IHDR/IDAT/IEND）与行滤波的宿主侧实现。
//
// 分工（与后端无关，写在这里是为了让三个 GPU 后端共用同一套容器语义）：
//   * 行滤波（filter / unfilter）是像素级计算，可由 GPU 完成后把整段 filtered
//     流交给 writePng()；
//   * deflate/inflate 走 zlib（CPU）：GPU 上的 deflate 在这些图像尺寸下没有收益，
//     且会让「码流是否与 libpng 兼容」变成一个不可验证的问题。
//
// 支持：8-bit、非隔行、color type 0/2/4/6（灰度/ RGB / 灰度+α / RGBA）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace png {

// PNG 行滤波类型（与规范一致）。
enum FilterType : int {
  kFilterNone = 0,
  kFilterSub = 1,
  kFilterUp = 2,
  kFilterAverage = 3,
  kFilterPaeth = 4,
};

// 参考实现：把一行原始样本按 filter 类型过滤到 out（out 不含 filter 字节）。
// prev 为上一行原始样本（第一行传 nullptr）；bpp = 每像素字节数。
void filterRow(int filter, const uint8_t *row, const uint8_t *prev, size_t n,
               int bpp, uint8_t *out);

// 参考实现：把一行 filtered 数据原地还原（row 既是输入也是输出）。
void unfilterRow(int filter, uint8_t *row, const uint8_t *prev, size_t n,
                 int bpp);

// 自适应滤波（逐行最小绝对值和）。返回 per-row filter 类型表。
std::vector<uint8_t> chooseAdaptiveFilters(const uint8_t *raw, int width,
                                           int height, int channels);

// 生成完整 filtered 流：每行 1 字节 filter + width*channels 字节数据。
// filters 为空时按 chooseAdaptiveFilters 选择；否则长度必须 == height。
std::vector<uint8_t> buildFilteredStream(const uint8_t *raw, int width,
                                         int height, int channels,
                                         const std::vector<uint8_t> &filters);

// 用已过滤的流写 PNG（filtered 长度必须 == height*(1 + width*channels)）。
// compression_level: 0..9（0 = 不压缩，仅测试用）。
bool writeFilteredPng(const uint8_t *filtered, size_t filtered_size, int width,
                      int height, int channels, int compression_level,
                      std::vector<uint8_t> &out, std::string *err);

// 读取 PNG（8-bit 非隔行 0/2/4/6）。成功返回 true。
bool readPng(const uint8_t *data, size_t size, ImageBuffer &img,
             std::string *err);

// 魔数判断。
inline bool hasPngSignature(const uint8_t *p, size_t n) {
  static const uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (n < 8) return false;
  for (int i = 0; i < 8; ++i) {
    if (p[i] != kSig[i]) return false;
  }
  return true;
}

} // namespace png
} // namespace cuda
} // namespace image_codec
