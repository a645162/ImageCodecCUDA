/*
 * File: bmp_io.cpp
 * Brief: Bmp io.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// bmp_io.cpp - 24-bit BMP 读写实现（CLI/测试用）。
#include "bmp_io.h"

#include <cstdio>
#include <cstring>

namespace image_codec {
namespace cuda {

namespace {

uint32_t rdU32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t rdU16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
void wrU16(uint8_t *p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>(v >> 8);
}
void wrU32(uint8_t *p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

} // namespace

int bmpLoad(const char *path, BmpImage &img) {
  img.width = 0;
  img.height = 0;
  img.bgr.clear();
  if (path == nullptr) return -1;

  FILE *f = std::fopen(path, "rb");
  if (f == nullptr) return -1;
  std::fseek(f, 0, SEEK_END);
  long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len < 54) {
    std::fclose(f);
    return -2;
  }
  std::vector<uint8_t> b(static_cast<size_t>(len));
  const size_t got = std::fread(b.data(), 1, b.size(), f);
  std::fclose(f);
  if (got != b.size() || b[0] != 'B' || b[1] != 'M') return -2;

  const uint32_t off_bits = rdU32(&b[10]);
  const int32_t w = static_cast<int32_t>(rdU32(&b[18]));
  const int32_t h = static_cast<int32_t>(rdU32(&b[22]));
  const uint16_t planes = rdU16(&b[26]);
  const uint16_t bpp = rdU16(&b[28]);
  const uint32_t comp = rdU32(&b[30]);
  const uint32_t header_size = rdU32(&b[14]);

  if (planes != 1 || bpp != 24 || comp != 0 || w <= 0 || h == 0) return -2;
  if (header_size < 40) return -2;

  const int abs_h = h < 0 ? -h : h;
  const bool bottom_up = h > 0;
  const size_t stride = (static_cast<size_t>(w) * 24 + 31) / 32 * 4;
  const size_t row_bytes = static_cast<size_t>(w) * 3;
  if (off_bits + static_cast<size_t>(abs_h - 1) * stride + row_bytes >
      b.size()) {
    return -2;
  }

  img.width = w;
  img.height = abs_h;
  img.bgr.assign(row_bytes * static_cast<size_t>(abs_h), 0);
  for (int r = 0; r < abs_h; ++r) {
    const size_t src =
        bottom_up ? static_cast<size_t>(abs_h - 1 - r) : static_cast<size_t>(r);
    std::memcpy(&img.bgr[static_cast<size_t>(r) * row_bytes],
                &b[off_bits + src * stride], row_bytes);
  }
  return 0;
}

int bmpSave(const char *path, const BmpImage &img) {
  if (path == nullptr || img.width <= 0 || img.height <= 0) return -1;
  if (img.bgr.size() < static_cast<size_t>(img.width) * img.height * 3) {
    return -1;
  }

  const size_t w = static_cast<size_t>(img.width);
  const size_t h = static_cast<size_t>(img.height);
  const size_t row_bytes = w * 3;
  const size_t stride = (w * 24 + 31) / 32 * 4;
  const uint32_t pixel_bytes = static_cast<uint32_t>(stride * h);
  const uint32_t file_size = 54 + pixel_bytes;

  std::vector<uint8_t> b(file_size, 0);
  b[0] = 'B';
  b[1] = 'M';
  wrU32(&b[2], file_size);
  wrU32(&b[10], 54); // off_bits
  wrU32(&b[14], 40); // INFOHEADER size
  wrU32(&b[18], static_cast<uint32_t>(img.width));
  wrU32(&b[22], static_cast<uint32_t>(img.height)); // 正数 = bottom-up
  wrU16(&b[26], 1);
  wrU16(&b[28], 24);
  wrU32(&b[30], 0); // BI_RGB
  wrU32(&b[34], pixel_bytes);
  wrU32(&b[38], 2835); // 水平分辨率 (72dpi)
  wrU32(&b[42], 2835); // 垂直分辨率
  wrU32(&b[46], 0);    // 调色板
  wrU32(&b[50], 0);    // 重要颜色

  for (size_t r = 0; r < h; ++r) {
    const size_t dst = (h - 1 - r) * stride; // bottom-up
    std::memcpy(&b[54 + dst], &img.bgr[r * row_bytes], row_bytes);
  }

  FILE *f = std::fopen(path, "wb");
  if (f == nullptr) return -1;
  const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
  std::fclose(f);
  return ok ? 0 : -2;
}

} // namespace cuda
} // namespace image_codec
