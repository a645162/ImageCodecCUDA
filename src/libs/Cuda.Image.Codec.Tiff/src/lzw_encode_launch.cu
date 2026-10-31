/*
 * File: lzw_encode_launch.cu
 * Brief: Lzw encode launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_encode_launch.cu - 绑定 kernel 启动参数（host 侧薄封装）。
#include "lzw_encode_launch.h"

namespace image_codec {
namespace cuda {
namespace tiff {

__global__ void lzw_encode_kernel(const uint8_t *__restrict__ input,
                                  uint32_t *__restrict__ dict,
                                  uint8_t *__restrict__ output,
                                  uint32_t *__restrict__ out_sizes,
                                  uint32_t unit_bytes, uint32_t max_unit,
                                  uint32_t n_units);

void launchLzwEncode(const uint8_t *input, uint32_t *dict, uint8_t *output,
                     uint32_t *out_sizes, uint32_t unit_bytes,
                     uint32_t max_unit, uint32_t n_units, int block_threads,
                     cudaStream_t stream) {
  if (n_units == 0) return;
  // 每个 unit 一个线程、寄存器与栈都吃得多，线程过多反而降频；裁剪到 [32,512]。
  int threads = block_threads > 0 ? block_threads : 128;
  if (threads < 32) threads = 32;
  if (threads > 512) threads = 512;
  const int blocks = static_cast<int>((n_units + threads - 1) / threads);
  lzw_encode_kernel<<<blocks, threads, 0, stream>>>(
      input, dict, output, out_sizes, unit_bytes, max_unit, n_units);
}

} // namespace tiff
} // namespace cuda
} // namespace image_codec
