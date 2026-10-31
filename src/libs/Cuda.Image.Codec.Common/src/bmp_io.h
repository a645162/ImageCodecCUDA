/*
 * File: bmp_io.h
 * Brief: Bmp io.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// bmp_io.h - 24-bit BMP 读写（CLI/测试用的中间格式，非产品路径）。
#pragma once

#include <cstdint>
#include <vector>

namespace image_codec {
namespace cuda {

struct BmpImage {
  int width = 0;
  int height = 0;           // 绝对值（正数）
  std::vector<uint8_t> bgr; // top-down BGR24（width*3 字节/行）
};

// 读取 24-bit BMP。成功返回 0；失败返回负值。
// 不支持的格式（非 24-bit / 压缩 / 位图掩码等）返回 -2。
int bmpLoad(const char *path, BmpImage &img);

// 写出 24-bit BMP（标准 bottom-up，BITMAPFILEHEADER+INFOHEADER+BGR 像素）。
// 成功返回 0。
int bmpSave(const char *path, const BmpImage &img);

} // namespace cuda
} // namespace image_codec
