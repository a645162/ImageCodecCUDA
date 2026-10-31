/*
 * File: png_container.cpp
 * Brief: Png container.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_container.cpp - PNG 容器与行滤波实现（zlib）。
#include "png_container.h"

#include <zlib.h>

#include <cstring>
#include <limits>

namespace image_codec {
namespace cuda {
namespace png {

namespace {

uint32_t readU32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

void appendU32(std::vector<uint8_t> &v, uint32_t x) {
  v.push_back(static_cast<uint8_t>((x >> 24) & 0xFF));
  v.push_back(static_cast<uint8_t>((x >> 16) & 0xFF));
  v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
  v.push_back(static_cast<uint8_t>(x & 0xFF));
}

// 追加一个 PNG chunk（长度 + 类型 + 数据 + CRC）。
void appendChunk(std::vector<uint8_t> &out, const char type[4],
                 const uint8_t *data, size_t size) {
  appendU32(out, static_cast<uint32_t>(size));
  const size_t type_pos = out.size();
  out.insert(out.end(), type, type + 4);
  if (size > 0 && data != nullptr) {
    out.insert(out.end(), data, data + size);
  }
  uLong crc = crc32(0L, Z_NULL, 0);
  crc = crc32(crc, out.data() + type_pos, 4 + static_cast<uInt>(size));
  appendU32(out, static_cast<uint32_t>(crc));
}

int colorTypeOf(int channels) {
  switch (channels) {
  case 1:
    return 0; // grayscale
  case 2:
    return 4; // gray + alpha
  case 3:
    return 2; // RGB
  case 4:
    return 6; // RGBA
  default:
    return -1;
  }
}

int channelsOfColorType(int color_type) {
  switch (color_type) {
  case 0:
    return 1;
  case 2:
    return 3;
  case 4:
    return 2; // gray + alpha（本库不把 alpha 铺成 4 通道，按 2 通道返回）
  case 6:
    return 4;
  default:
    return -1;
  }
}

int paeth(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = p > a ? p - a : a - p;
  const int pb = p > b ? p - b : b - p;
  const int pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

} // namespace

void filterRow(int filter, const uint8_t *row, const uint8_t *prev, size_t n,
               int bpp, uint8_t *out) {
  const size_t b = static_cast<size_t>(bpp);
  switch (filter) {
  case kFilterNone:
    std::memcpy(out, row, n);
    break;
  case kFilterSub:
    for (size_t i = 0; i < n; ++i) {
      const int left = i >= b ? row[i - b] : 0;
      out[i] = static_cast<uint8_t>(row[i] - left);
    }
    break;
  case kFilterUp:
    for (size_t i = 0; i < n; ++i) {
      const int up = prev != nullptr ? prev[i] : 0;
      out[i] = static_cast<uint8_t>(row[i] - up);
    }
    break;
  case kFilterAverage:
    for (size_t i = 0; i < n; ++i) {
      const int left = i >= b ? row[i - b] : 0;
      const int up = prev != nullptr ? prev[i] : 0;
      out[i] = static_cast<uint8_t>(row[i] - ((left + up) >> 1));
    }
    break;
  case kFilterPaeth:
    for (size_t i = 0; i < n; ++i) {
      const int left = i >= b ? row[i - b] : 0;
      const int up = prev != nullptr ? prev[i] : 0;
      const int ul = (i >= b && prev != nullptr) ? prev[i - b] : 0;
      out[i] = static_cast<uint8_t>(row[i] - paeth(left, up, ul));
    }
    break;
  default:
    std::memcpy(out, row, n);
    break;
  }
}

void unfilterRow(int filter, uint8_t *row, const uint8_t *prev, size_t n,
                 int bpp) {
  const size_t b = static_cast<size_t>(bpp);
  switch (filter) {
  case kFilterNone:
    break;
  case kFilterSub:
    for (size_t i = b; i < n; ++i) {
      row[i] = static_cast<uint8_t>(row[i] + row[i - b]);
    }
    break;
  case kFilterUp:
    if (prev != nullptr) {
      for (size_t i = 0; i < n; ++i) {
        row[i] = static_cast<uint8_t>(row[i] + prev[i]);
      }
    }
    break;
  case kFilterAverage:
    for (size_t i = 0; i < n; ++i) {
      const int left = i >= b ? row[i - b] : 0;
      const int up = prev != nullptr ? prev[i] : 0;
      row[i] = static_cast<uint8_t>(row[i] + ((left + up) >> 1));
    }
    break;
  case kFilterPaeth:
    for (size_t i = 0; i < n; ++i) {
      const int left = i >= b ? row[i - b] : 0;
      const int up = prev != nullptr ? prev[i] : 0;
      const int ul = (i >= b && prev != nullptr) ? prev[i - b] : 0;
      row[i] = static_cast<uint8_t>(row[i] + paeth(left, up, ul));
    }
    break;
  default:
    // 非法 filter：按 None 处理（与 libpng 的宽容行为一致）。
    break;
  }
}

std::vector<uint8_t> chooseAdaptiveFilters(const uint8_t *raw, int width,
                                           int height, int channels) {
  std::vector<uint8_t> filters;
  if (raw == nullptr || width <= 0 || height <= 0 || channels <= 0) {
    return filters;
  }
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(channels);
  const size_t stride = n;
  filters.reserve(static_cast<size_t>(height));
  std::vector<uint8_t> candidate(n);
  for (int y = 0; y < height; ++y) {
    const uint8_t *row = raw + static_cast<size_t>(y) * stride;
    const uint8_t *prev = y > 0 ? row - stride : nullptr;
    uint64_t best_sum = std::numeric_limits<uint64_t>::max();
    int best = kFilterNone;
    for (int f = kFilterNone; f <= kFilterPaeth; ++f) {
      filterRow(f, row, prev, n, channels, candidate.data());
      uint64_t sum = 0;
      for (size_t i = 0; i < n; ++i) {
        const int v = static_cast<int8_t>(candidate[i]);
        sum += static_cast<uint64_t>(v < 0 ? -v : v);
      }
      if (sum < best_sum) {
        best_sum = sum;
        best = f;
      }
    }
    filters.push_back(static_cast<uint8_t>(best));
  }
  return filters;
}

std::vector<uint8_t> buildFilteredStream(const uint8_t *raw, int width,
                                         int height, int channels,
                                         const std::vector<uint8_t> &filters) {
  std::vector<uint8_t> out;
  if (raw == nullptr || width <= 0 || height <= 0 || channels <= 0) {
    return out;
  }
  if (!filters.empty() && filters.size() != static_cast<size_t>(height)) {
    return out;
  }
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(channels);
  const size_t stride = n;
  out.resize(static_cast<size_t>(height) * (n + 1));
  for (int y = 0; y < height; ++y) {
    const uint8_t *row = raw + static_cast<size_t>(y) * stride;
    const uint8_t *prev = y > 0 ? row - stride : nullptr;
    const int f = filters.empty()
                      ? kFilterNone
                      : static_cast<int>(filters[static_cast<size_t>(y)]);
    uint8_t *dst = out.data() + static_cast<size_t>(y) * (n + 1);
    dst[0] = static_cast<uint8_t>(f);
    filterRow(f, row, prev, n, channels, dst + 1);
  }
  return out;
}

bool writeFilteredPng(const uint8_t *filtered, size_t filtered_size, int width,
                      int height, int channels, int compression_level,
                      std::vector<uint8_t> &out, std::string *err) {
  out.clear();
  const int color_type = colorTypeOf(channels);
  if (color_type < 0 || width <= 0 || height <= 0) {
    if (err != nullptr) *err = "PNG: unsupported channel count / size";
    return false;
  }
  const size_t expected =
      static_cast<size_t>(height) *
      (1 + static_cast<size_t>(width) * static_cast<size_t>(channels));
  if (filtered == nullptr || filtered_size != expected) {
    if (err != nullptr) *err = "PNG: filtered stream size mismatch";
    return false;
  }

  // zlib 压缩整段 filtered 流。
  uLongf bound = compressBound(static_cast<uLong>(filtered_size));
  std::vector<uint8_t> compressed(bound);
  const int level = compression_level < 0
                        ? 6
                        : (compression_level > 9 ? 9 : compression_level);
  const int rc = compress2(compressed.data(), &bound, filtered,
                           static_cast<uLong>(filtered_size), level);
  if (rc != Z_OK) {
    if (err != nullptr) *err = "PNG: zlib compress failed";
    return false;
  }
  compressed.resize(bound);

  static const uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  out.insert(out.end(), kSig, kSig + 8);

  uint8_t ihdr[13];
  ihdr[0] = static_cast<uint8_t>((width >> 24) & 0xFF);
  ihdr[1] = static_cast<uint8_t>((width >> 16) & 0xFF);
  ihdr[2] = static_cast<uint8_t>((width >> 8) & 0xFF);
  ihdr[3] = static_cast<uint8_t>(width & 0xFF);
  ihdr[4] = static_cast<uint8_t>((height >> 24) & 0xFF);
  ihdr[5] = static_cast<uint8_t>((height >> 16) & 0xFF);
  ihdr[6] = static_cast<uint8_t>((height >> 8) & 0xFF);
  ihdr[7] = static_cast<uint8_t>(height & 0xFF);
  ihdr[8] = 8;                                // bit depth
  ihdr[9] = static_cast<uint8_t>(color_type); // color type
  ihdr[10] = 0;                               // compression
  ihdr[11] = 0;                               // filter method
  ihdr[12] = 0;                               // interlace: none
  appendChunk(out, "IHDR", ihdr, sizeof(ihdr));

  // 单个 IDAT 足够；超大图由 zlib 自身窗口处理，不依赖 chunk 拆分。
  appendChunk(out, "IDAT", compressed.data(), compressed.size());
  appendChunk(out, "IEND", nullptr, 0);
  return true;
}

bool readPng(const uint8_t *data, size_t size, ImageBuffer &img,
             std::string *err) {
  if (data == nullptr || size < 8 || !hasPngSignature(data, size)) {
    if (err != nullptr) *err = "PNG: bad signature";
    return false;
  }
  size_t pos = 8;
  int width = 0;
  int height = 0;
  int bit_depth = 0;
  int color_type = -1;
  int interlace = 0;
  std::vector<uint8_t> idat;

  while (pos + 8 <= size) {
    const uint32_t len = readU32(data + pos);
    const char *type = reinterpret_cast<const char *>(data + pos + 4);
    if (pos + 8 + len + 4 > size) {
      if (err != nullptr) *err = "PNG: truncated chunk";
      return false;
    }
    const uint8_t *payload = data + pos + 8;
    if (std::strncmp(type, "IHDR", 4) == 0 && len >= 13) {
      width = static_cast<int>(readU32(payload));
      height = static_cast<int>(readU32(payload + 4));
      bit_depth = payload[8];
      color_type = payload[9];
      interlace = payload[12];
    } else if (std::strncmp(type, "IDAT", 4) == 0) {
      idat.insert(idat.end(), payload, payload + len);
    } else if (std::strncmp(type, "IEND", 4) == 0) {
      break;
    }
    pos += 8 + len + 4;
  }

  const int channels = channelsOfColorType(color_type);
  if (width <= 0 || height <= 0 || channels <= 0 || bit_depth != 8 ||
      interlace != 0) {
    if (err != nullptr) {
      *err = "PNG: only 8-bit non-interlaced color types 0/2/4/6 are supported";
    }
    return false;
  }
  if (idat.empty()) {
    if (err != nullptr) *err = "PNG: no IDAT";
    return false;
  }

  const size_t row_bytes =
      static_cast<size_t>(width) * static_cast<size_t>(channels);
  const size_t expected = (row_bytes + 1) * static_cast<size_t>(height);
  std::vector<uint8_t> raw(expected);

  // 流式 inflate：expected 已知，但健壮起见仍按 z_stream 驱动。
  z_stream zs{};
  if (inflateInit(&zs) != Z_OK) {
    if (err != nullptr) *err = "PNG: inflateInit failed";
    return false;
  }
  zs.next_in = const_cast<Bytef *>(idat.data());
  zs.avail_in = static_cast<uInt>(idat.size());
  zs.next_out = raw.data();
  zs.avail_out = static_cast<uInt>(raw.size());
  int rc = inflate(&zs, Z_FINISH);
  const size_t produced = raw.size() - zs.avail_out;
  inflateEnd(&zs);
  if (rc != Z_STREAM_END || produced != raw.size()) {
    if (err != nullptr) *err = "PNG: inflate failed or size mismatch";
    return false;
  }

  if (!img.allocate(width, height, channels)) {
    if (err != nullptr) *err = "PNG: allocation failed";
    return false;
  }
  for (int y = 0; y < height; ++y) {
    uint8_t *src = raw.data() + static_cast<size_t>(y) * (row_bytes + 1);
    const int filter = src[0];
    uint8_t *dst = img.data.data() + static_cast<size_t>(y) * row_bytes;
    const uint8_t *prev =
        y > 0 ? img.data.data() + static_cast<size_t>(y - 1) * row_bytes
              : nullptr;
    std::memcpy(dst, src + 1, row_bytes);
    unfilterRow(filter, dst, prev, row_bytes, channels);
  }
  return true;
}

} // namespace png
} // namespace cuda
} // namespace image_codec
