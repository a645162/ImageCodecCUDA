/*
 * File: metrics_launch.h
 * Brief: Metrics launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics_launch.h - 指标 kernel 的启动入口（.cpp 侧不写 __global__ 语法）。
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace metrics {

// 处理一个 band。a/b 是设备上「从整图 y0-halo 行开始、共 rows 行」的缓冲，
// weights 为 11 个归一化高斯权重。acc_* 为设备上的累加器（调用方负责清零）。
//
// with_window == 0 时跳过 11x11 窗，只累加平方误差：这是 MSE/PSNR 的路径，
// 内存流量比 SSIM 低两个数量级。
void launchMetricsBand(const uint8_t *a, const uint8_t *b, int width,
                       int height, int channels, int y0, int band_rows,
                       int halo, int rows, int with_window,
                       const double *weights, double *acc_ssim, double *acc_cs,
                       double *acc_sq, cudaStream_t stream);

} // namespace metrics
} // namespace cuda
} // namespace image_codec
