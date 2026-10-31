/*
 * File: png_filter_launch.h
 * Brief: Png filter launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_filter_launch.h - PNG 滤波 kernel 启动入口。
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace png {

// 行滤波。types == nullptr 时使用 uniform_type（0..4）。
// block_threads <= 0 时用默认值（256）；上限 1024（block 内最多 32 个 warp，
// 与 png_filter_sums_kernel 的 smem[5][32] 归约布局对应）。
void launchPngFilter(const uint8_t *src, uint8_t *dst, const uint8_t *types,
                     int width, int height, int channels, int uniform_type,
                     int block_threads, cudaStream_t stream);

// 自适应统计：输出 sums[height * 5]。
void launchPngFilterSums(const uint8_t *src, uint64_t *sums, int width,
                         int height, int channels, int block_threads,
                         cudaStream_t stream);

} // namespace png
} // namespace cuda
} // namespace image_codec
