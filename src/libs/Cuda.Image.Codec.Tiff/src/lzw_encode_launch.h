/*
 * File: lzw_encode_launch.h
 * Brief: Lzw encode launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_encode_launch.h - kernel 启动入口（.cpp 侧不直接依赖 __global__ 语法）。
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace tiff {

// 每个 unit 一个线程；input/output/sizes 均为 device 指针。
// block_threads <= 0 时用默认值（128），内部裁剪到 [32,512]。
void launchLzwEncode(const uint8_t *input, uint32_t *dict, uint8_t *output,
                     uint32_t *out_sizes, uint32_t unit_bytes,
                     uint32_t max_unit, uint32_t n_units, int block_threads,
                     cudaStream_t stream);

} // namespace tiff
} // namespace cuda
} // namespace image_codec
