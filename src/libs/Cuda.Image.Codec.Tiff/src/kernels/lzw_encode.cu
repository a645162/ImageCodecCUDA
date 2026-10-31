/*
 * File: lzw_encode.cu
 * Brief: Lzw encode.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// kernels/lzw_encode.cu - TIFF LZW 编码 kernel（单元并行）。
//
// 每个线程独立编码一条 strip 或一个 tile，使用全局内存里的开放寻址哈希字典
// （槽位打包为 ((key+1)<<12)|code，一次探测一次内存访问）。
// 码宽增长 9->10->11->12 于 next_code == 512/1024/2048 时切换，满字典(4096)
// 发 ClearCode(256) 并重置，EOI(257) 结束——与 Common/lzw_codec.cpp 完全一致。
//
// 单位内串行、单位间并行：tile 布局（256/512）天然提供成千上万个并行单元，
// 这也是本后端推荐 tiled TIFF 的原因（strip 太少时 GPU 没有并行度可用）。
#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace tiff {

__device__ __forceinline__ uint32_t lzwHash(uint32_t key) {
  return ((key * 2654435761u) >> 20) & 4095u;
}

__device__ __forceinline__ void putBits(uint8_t *out, uint32_t *out_len,
                                        uint32_t *bitbuf, int *nbits,
                                        uint32_t value, int width) {
  *bitbuf = (*bitbuf << width) | value;
  *nbits += width;
  while (*nbits >= 8) {
    *nbits -= 8;
    out[(*out_len)++] = static_cast<uint8_t>(*bitbuf >> *nbits);
  }
  *bitbuf &= (1u << *nbits) - 1u;
}

__global__ void lzw_encode_kernel(const uint8_t *__restrict__ input,
                                  uint32_t *__restrict__ dict,
                                  uint8_t *__restrict__ output,
                                  uint32_t *__restrict__ out_sizes,
                                  uint32_t unit_bytes, uint32_t max_unit,
                                  uint32_t n_units) {
  const uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= n_units) return;

  const uint8_t *in = input + static_cast<size_t>(unit) * unit_bytes;
  uint8_t *out = output + static_cast<size_t>(unit) * max_unit;
  uint32_t *d = dict + static_cast<size_t>(unit) * 4096u;

  // 字典清零：4096 个槽位，由本线程负责（单位间不共享）。
  for (uint32_t i = 0; i < 4096u; ++i) {
    d[i] = 0u;
  }

  uint32_t bitbuf = 0;
  int nbits = 0;
  uint32_t out_len = 0;
  uint32_t next_code = 258u;
  int width = 9;

  putBits(out, &out_len, &bitbuf, &nbits, 256u, 9); // ClearCode

  if (unit_bytes > 0) {
    uint32_t prefix = in[0];
    for (uint32_t i = 1; i < unit_bytes; ++i) {
      const uint32_t byte = in[i];
      const uint32_t key = (prefix << 8) | byte;
      uint32_t h = lzwHash(key);
      uint32_t e = d[h];
      while (e != 0u && (e >> 12) != key + 1u) {
        h = (h + 1u) & 4095u;
        e = d[h];
      }
      if ((e >> 12) == key + 1u) {
        prefix = e & 4095u;
      } else {
        putBits(out, &out_len, &bitbuf, &nbits, prefix, width);
        if (next_code < 4096u) {
          d[h] = ((key + 1u) << 12) | next_code;
          next_code++;
          if (next_code == 512u) {
            width = 10;
          } else if (next_code == 1024u) {
            width = 11;
          } else if (next_code == 2048u) {
            width = 12;
          }
        } else {
          putBits(out, &out_len, &bitbuf, &nbits, 256u, width);
          for (uint32_t j = 0; j < 4096u; ++j) {
            d[j] = 0u;
          }
          next_code = 258u;
          width = 9;
        }
        prefix = byte;
      }
    }
    putBits(out, &out_len, &bitbuf, &nbits, prefix, width);
    // 最后一个数据码之后补一次「假想建表」的宽度记账：解码器读到最后一个码
    // 仍会建一条（下一条才用得到的）条目；若这次建表正好跨过 512/1024/2048，
    // 它会按更宽的码读 EOI，编码器不补这一步就会位流错位（缺 EOI / 多解字节）。
    if (next_code < 4096u) {
      next_code++;
      if (next_code == 512u) width = 10;
      else if (next_code == 1024u) width = 11;
      else if (next_code == 2048u) width = 12;
    }
  }
  putBits(out, &out_len, &bitbuf, &nbits, 257u, width); // EOI
  if (nbits > 0) {
    out[out_len++] = static_cast<uint8_t>(bitbuf << (8 - nbits));
  }
  out_sizes[unit] = out_len;
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
