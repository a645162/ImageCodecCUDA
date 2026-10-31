/*
 * File: png_filter.cu
 * Brief: Png filter.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// kernels/png_filter.cu - PNG 行滤波 kernel。
//
// 关键性质：PNG 的 filter 预测值只来自「原始行」的左侧/B 上方/左上样本，
// 不依赖同一行已过滤的输出，因此每一行、每一字节都可以完全并行。
// 这正是 PNG 编码里唯一值得放到 GPU 的阶段；deflate 留在 zlib。
#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace png {

__device__ __forceinline__ int paethPredict(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = p > a ? p - a : a - p;
  const int pb = p > b ? p - b : b - p;
  const int pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

// 单行滤波：blockIdx.x = 行号，线程按字节跨步。
// types 为每行 filter 类型数组；uniform_type >= 0 时忽略 types。
__global__ void png_filter_rows_kernel(const uint8_t *__restrict__ src,
                                       uint8_t *__restrict__ dst,
                                       const uint8_t *__restrict__ types,
                                       int width, int channels,
                                       int uniform_type) {
  const int y = blockIdx.x;
  const int row_bytes = width * channels;
  const int bpp = channels;
  const uint8_t *row = src + static_cast<size_t>(y) * row_bytes;
  const uint8_t *prev = y > 0 ? row - row_bytes : nullptr;
  uint8_t *out = dst + static_cast<size_t>(y) * (row_bytes + 1);

  const int filter =
      uniform_type >= 0 ? uniform_type : static_cast<int>(types[y]);
  if (threadIdx.x == 0) {
    out[0] = static_cast<uint8_t>(filter);
  }
  for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    const int left = i >= bpp ? row[i - bpp] : 0;
    const int up = prev != nullptr ? prev[i] : 0;
    const int ul = (i >= bpp && prev != nullptr) ? prev[i - bpp] : 0;
    int pred = 0;
    switch (filter) {
    case 1: // Sub
      pred = left;
      break;
    case 2: // Up
      pred = up;
      break;
    case 3: // Average
      pred = (left + up) >> 1;
      break;
    case 4: // Paeth
      pred = paethPredict(left, up, ul);
      break;
    default: // None
      pred = 0;
      break;
    }
    out[1 + i] = static_cast<uint8_t>(row[i] - pred);
  }
}

// 自适应滤波：为每行计算 5 种 filter 的残差绝对值和。
// 一个 block 负责一行，块内归约后写入 sums[y*5 + f]。
__global__ void png_filter_sums_kernel(const uint8_t *__restrict__ src,
                                       uint64_t *__restrict__ sums, int width,
                                       int channels) {
  const int y = blockIdx.x;
  const int row_bytes = width * channels;
  const int bpp = channels;
  const uint8_t *row = src + static_cast<size_t>(y) * row_bytes;
  const uint8_t *prev = y > 0 ? row - row_bytes : nullptr;

  uint64_t acc[5] = {0, 0, 0, 0, 0};
  for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    const int left = i >= bpp ? row[i - bpp] : 0;
    const int up = prev != nullptr ? prev[i] : 0;
    const int ul = (i >= bpp && prev != nullptr) ? prev[i - bpp] : 0;
    const int v0 = row[i];
    const int preds[4] = {left, up, (left + up) >> 1,
                          paethPredict(left, up, ul)};
    {
      const int r = v0;
      acc[0] += static_cast<uint64_t>(r < 0 ? -r : r);
    }
    for (int f = 0; f < 4; ++f) {
      const int r = v0 - preds[f];
      acc[f + 1] += static_cast<uint64_t>(r < 0 ? -r : r);
    }
  }

  __shared__ uint64_t smem[5][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  for (int f = 0; f < 5; ++f) {
    // warp 内归约
    uint64_t v = acc[f];
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    if (lane == 0) smem[f][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < 5) {
    uint64_t v = 0;
    for (int w = 0; w < warps; ++w) v += smem[threadIdx.x][w];
    sums[static_cast<size_t>(y) * 5 + threadIdx.x] = v;
  }
}

} // namespace png
} // namespace cuda
} // namespace image_codec
