/*
 * File: png_filter_launch.cu
 * Brief: Png filter launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_filter_launch.cu - 绑定 PNG 滤波 kernel 启动参数。
#include "png_filter_launch.h"

namespace image_codec {
namespace cuda {
namespace png {

__global__ void png_filter_rows_kernel(const uint8_t *__restrict__ src,
                                       uint8_t *__restrict__ dst,
                                       const uint8_t *__restrict__ types,
                                       int width, int channels,
                                       int uniform_type);

__global__ void png_filter_sums_kernel(const uint8_t *__restrict__ src,
                                       uint64_t *__restrict__ sums, int width,
                                       int channels);

namespace {

// 线程数裁剪：[32, 1024]，并按 32（warp）对齐——归约依赖整 warp 参与。
int clampThreads(int block_threads) {
  int t = block_threads > 0 ? block_threads : 256;
  if (t < 32) t = 32;
  if (t > 1024) t = 1024;
  return (t / 32) * 32;
}

} // namespace

void launchPngFilter(const uint8_t *src, uint8_t *dst, const uint8_t *types,
                     int width, int height, int channels, int uniform_type,
                     int block_threads, cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  // 默认 256 线程/行：行字节数在 768..48K 之间时摊销良好；更大行数也由跨步循环覆盖。
  png_filter_rows_kernel<<<height, clampThreads(block_threads), 0, stream>>>(
      src, dst, types, width, channels, uniform_type);
}

void launchPngFilterSums(const uint8_t *src, uint64_t *sums, int width,
                         int height, int channels, int block_threads,
                         cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  png_filter_sums_kernel<<<height, clampThreads(block_threads), 0, stream>>>(
      src, sums, width, channels);
}

} // namespace png
} // namespace cuda
} // namespace image_codec
