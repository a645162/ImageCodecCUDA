/*
 * File: metrics_launch.cu
 * Brief: Metrics launch.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics_launch.cu - 绑定指标 kernel 的启动参数。
#include "metrics_launch.h"

namespace image_codec {
namespace cuda {
namespace metrics {

struct GaussWeights11 {
  double w[11];
};

__global__ void metrics_band_kernel(const uint8_t *__restrict__ a,
                                    const uint8_t *__restrict__ b, int width,
                                    int height, int channels, int y0,
                                    int band_rows, int halo, int rows,
                                    int with_window, GaussWeights11 weights,
                                    double *acc_ssim, double *acc_cs,
                                    double *acc_sq);

void launchMetricsBand(const uint8_t *a, const uint8_t *b, int width,
                       int height, int channels, int y0, int band_rows,
                       int halo, int rows, int with_window,
                       const double *weights, double *acc_ssim, double *acc_cs,
                       double *acc_sq, cudaStream_t stream) {
  if (width <= 0 || height <= 0 || band_rows <= 0 || rows <= 0) return;

  // 固定 256 线程：metrics_band_kernel 的共享内存归约按 256 写死。
  // 这个 kernel 是统计型的，block 大小对吞吐不敏感（瓶颈在 11x11 窗的读）。
  constexpr int kThreads = 256;

  GaussWeights11 w{};
  for (int i = 0; i < 11; ++i) w.w[i] = weights[i];

  const long long total = static_cast<long long>(band_rows) * width * channels;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;

  metrics_band_kernel<<<blocks, kThreads, 0, stream>>>(
      a, b, width, height, channels, y0, band_rows, halo, rows, with_window, w,
      acc_ssim, acc_cs, acc_sq);
}

} // namespace metrics
} // namespace cuda
} // namespace image_codec
