/*
 * File: lzw_codec.cpp
 * Brief: Lzw codec.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_codec.cpp - TIFF LZW CPU 参考实现。
#include "lzw_codec.h"

#include <algorithm>

namespace image_codec {
namespace cuda {
namespace lzw {

namespace {

constexpr uint16_t kClear = 256;
constexpr uint16_t kEoi = 257;
constexpr uint16_t kFirst = 258;   // 第一个空闲码
constexpr uint16_t kMaxEnt = 4096; // 字典容量
constexpr uint16_t kNoPrefix = 0xFFFF;

inline void putBits(std::vector<uint8_t> &out, uint32_t &bitbuf, int &nbits,
                    uint32_t value, int width) {
  bitbuf = (bitbuf << width) | value;
  nbits += width;
  while (nbits >= 8) {
    nbits -= 8;
    out.push_back(static_cast<uint8_t>(bitbuf >> nbits));
  }
  bitbuf &= (1u << nbits) - 1u;
}

inline uint32_t hashKey(uint32_t key) {
  return ((key * 2654435761u) >> 20) & 4095u;
}

} // namespace

int encode(const uint8_t *in, size_t in_size, std::vector<uint8_t> &out) {
  out.clear();
  if (in == nullptr && in_size > 0) return -1;
  // 字典槽位打包为 ((key+1)<<12) | code：一次探测一次内存访问。
  std::vector<uint32_t> dict(4096, 0);
  uint32_t bitbuf = 0;
  int nbits = 0;
  uint32_t next_code = kFirst;
  int width = 9;

  putBits(out, bitbuf, nbits, kClear, 9);
  if (in_size > 0) {
    uint32_t prefix = in[0];
    for (size_t i = 1; i < in_size; ++i) {
      const uint32_t byte = in[i];
      const uint32_t key = (prefix << 8) | byte;
      uint32_t h = hashKey(key);
      uint32_t e = dict[h];
      while (e != 0 && (e >> 12) != key + 1) {
        h = (h + 1) & 4095u;
        e = dict[h];
      }
      if ((e >> 12) == key + 1) {
        prefix = e & 4095u;
      } else {
        putBits(out, bitbuf, nbits, prefix, width);
        if (next_code < kMaxEnt) {
          dict[h] = ((key + 1) << 12) | next_code;
          next_code++;
          if (next_code == 512) {
            width = 10;
          } else if (next_code == 1024) {
            width = 11;
          } else if (next_code == 2048) {
            width = 12;
          }
        } else {
          putBits(out, bitbuf, nbits, kClear, width);
          std::fill(dict.begin(), dict.end(), 0u);
          next_code = kFirst;
          width = 9;
        }
        prefix = byte;
      }
    }
    putBits(out, bitbuf, nbits, prefix, width);
    // 最后一个数据码之后补一次「假想建表」的宽度记账：解码器读到最后一个码
    // 仍会建一条（下一条才用得到的）条目；若这次建表正好跨过 512/1024/2048，
    // 它会按更宽的码读 EOI，编码器不补这一步就会位流错位（缺 EOI / 多解字节）。
    if (next_code < 4096) {
      next_code++;
      if (next_code == 512) width = 10;
      else if (next_code == 1024) width = 11;
      else if (next_code == 2048) width = 12;
    }
  }
  putBits(out, bitbuf, nbits, kEoi, width);
  if (nbits > 0) {
    out.push_back(static_cast<uint8_t>(bitbuf << (8 - nbits)));
  }
  return 0;
}

int decode(const uint8_t *in, size_t in_size, std::vector<uint8_t> &out) {
  out.clear();
  if (in == nullptr && in_size > 0) return -1;

  // 前缀链：code -> (prefix_code, suffix_byte)，输出时沿链收集再反转。
  uint16_t prefix[kMaxEnt];
  uint8_t suffix[kMaxEnt];
  for (uint16_t i = 0; i < 256; ++i) {
    prefix[i] = kNoPrefix;
    suffix[i] = static_cast<uint8_t>(i);
  }

  uint32_t nextdata = 0;
  int nextbits = 0;
  size_t in_pos = 0;
  int nbits = 9;
  int next_code = kFirst;
  int old = -1;

  // 单条字符串最长不超过字典条目数（4096），留余量。
  uint8_t stack[kMaxEnt + 32];

  auto getNextCode = [&]() -> int {
    while (nextbits < nbits) {
      if (in_pos >= in_size) return -1; // 数据不足
      nextdata = (nextdata << 8) | in[in_pos++];
      nextbits += 8;
    }
    const int code = static_cast<int>((nextdata >> (nextbits - nbits)) &
                                      ((1u << nbits) - 1u));
    nextbits -= nbits;
    return code;
  };

  auto widthUp = [&]() {
    if (next_code == 511) {
      nbits = 10;
    } else if (next_code == 1023) {
      nbits = 11;
    } else if (next_code == 2047) {
      nbits = 12;
    }
  };

  for (;;) {
    const int code = getNextCode();
    if (code < 0) return -2; // 位流提前结束（缺 EOI）

    if (code == kEoi) break;
    if (code == kClear) {
      next_code = kFirst;
      nbits = 9;
      old = -1;
      continue;
    }
    if (code >= kMaxEnt) return -3;

    if (old == -1) {
      // ClearCode 之后第一个码必须是字面量。
      if (code >= 256) return -4;
      out.push_back(static_cast<uint8_t>(code));
      old = code;
      continue;
    }

    size_t sp = 0;
    if (code < next_code) {
      int c = code;
      while (prefix[c] != kNoPrefix) {
        stack[sp++] = suffix[c];
        c = prefix[c];
      }
      stack[sp++] = suffix[c];
      const uint8_t first_byte = stack[sp - 1];
      for (size_t i = sp; i-- > 0;) out.push_back(stack[i]);
      if (next_code < kMaxEnt) {
        prefix[next_code] = static_cast<uint16_t>(old);
        suffix[next_code] = first_byte;
        next_code++;
        widthUp();
      }
    } else if (code == next_code) {
      // KwKwK：新条目 = dict[old] + first(dict[old])。
      int c = old;
      while (prefix[c] != kNoPrefix) {
        stack[sp++] = suffix[c];
        c = prefix[c];
      }
      stack[sp++] = suffix[c];
      const uint8_t first_byte = stack[sp - 1];
      for (size_t i = sp; i-- > 0;) out.push_back(stack[i]);
      out.push_back(first_byte);
      if (next_code < kMaxEnt) {
        prefix[next_code] = static_cast<uint16_t>(old);
        suffix[next_code] = first_byte;
        next_code++;
        widthUp();
      }
    } else {
      return -5; // 非法码（code > next_code）
    }
    old = code;
  }
  return 0;
}

} // namespace lzw
} // namespace cuda
} // namespace image_codec
