/*
 * File: metrics.cu
 * Brief: Metrics.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics.cu - MSE / SSIM / MS-SSIM 的 CUDA kernel（单尺度累加）。
//
// 并行形态：一个线程负责一个 (像素, 通道)。
//   * with_window == 0（只要 MSE/PSNR）：只做减法-平方，每像素 2 次读，
//     不碰 11x11 窗；
//   * with_window != 0（SSIM / MS-SSIM）：在自己的 11x11 高斯窗上直接求和。
//
// 为什么不做「先可分离模糊成 5 张平面再合成」：那需要 5 张与图等大的 float 平面
// （32000x19656 的图就是 12.5 GB 显存），而我们这里连整幅图都不需要上传——
// 按行 band 上传即可。直方窗口的代价是每像素 2x121 次读，全部命中 L2/纹理路径，
// 实测远低于一次 12.5 GB 的设备往返。
//
// 边界：复制边缘（replicate），与 metrics_cpu.cpp 的参考实现完全一致。
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace image_codec {
namespace cuda {
namespace metrics {

// 权重用 double：CPU 参考实现用 double 计算，kernel 若用 float，11x11 窗的
// 加权和会带上 1e-8 的相对误差，均值化之后两边的 SSIM 就差在 1e-6 量级——
// 虽然很小，但「GPU 与 CPU 结果一致」的断言就只能靠放宽容差来通过，那不是对的。
struct GaussWeights11 {
  double w[11];
};

namespace {

// 单个 (像素, 通道) 的高斯窗统计。逻辑与 CPU 参考实现逐项对应，
// 改这里必须同步改 metrics_cpu.cpp，否则两边数字会对不上。
__device__ inline void accumulateWindow(const uint8_t *__restrict__ a,
                                        const uint8_t *__restrict__ b,
                                        int width, int height, int channels,
                                        int gy, int x, int c, int top, int halo,
                                        const GaussWeights11 &weights,
                                        double *out_ssim, double *out_cs) {
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;

  for (int dy = -halo; dy <= halo; ++dy) {
    const int syi = gy + dy;
    const int cy = syi < 0 ? 0 : (syi >= height ? height - 1 : syi);
    const int li = cy - top; // 本 band 缓冲内的行号（host 已按同样规则填充）

    const double wy = weights.w[dy + halo];
    const size_t row_base =
        static_cast<size_t>(li) * static_cast<size_t>(width) * channels;

    for (int dx = -halo; dx <= halo; ++dx) {
      const int sxi = x + dx;
      const int cx = sxi < 0 ? 0 : (sxi >= width ? width - 1 : sxi);

      const size_t o = row_base +
                       static_cast<size_t>(cx) * static_cast<size_t>(channels) +
                       c;
      const double px = static_cast<double>(a[o]);
      const double py = static_cast<double>(b[o]);
      const double g = wy * weights.w[dx + halo];

      sx += g * px;
      sy += g * py;
      sxx += g * px * px;
      syy += g * py * py;
      sxy += g * px * py;
    }
  }

  constexpr double kC1 = (0.01 * 255.0) * (0.01 * 255.0);
  constexpr double kC2 = (0.03 * 255.0) * (0.03 * 255.0);

  const double vx = sxx - sx * sx;
  const double vy = syy - sy * sy;
  const double vxy = sxy - sx * sy;

  const double lum = (2.0 * sx * sy + kC1) / (sx * sx + sy * sy + kC1);
  const double den = vx + vy + kC2;
  const double cs = den > 0.0 ? (2.0 * vxy + kC2) / den : 0.0;

  *out_cs = cs;
  *out_ssim = lum * cs;
}

} // namespace

// 归约结果累加到全局计数器（double 原子加在 sm_60+ 可用；本库架构下限 sm_61/75）。
__global__ void metrics_band_kernel(const uint8_t *__restrict__ a,
                                    const uint8_t *__restrict__ b, int width,
                                    int height, int channels, int y0,
                                    int band_rows, int halo, int rows,
                                    int with_window, GaussWeights11 weights,
                                    double *acc_ssim, double *acc_cs,
                                    double *acc_sq) {
  const int tiles = width * channels;
  const long long total = static_cast<long long>(band_rows) * tiles;

  double local_ssim = 0.0;
  double local_cs = 0.0;
  double local_sq = 0.0;

  const int top = y0 - halo;

  for (long long index =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int gy = y0 + static_cast<int>(index / tiles); // 整图行号
    const int rem = static_cast<int>(index % tiles);
    const int x = rem / channels;
    const int c = rem % channels;

    // band 缓冲的第 0 行对应整图 y0-halo 行（host 按同一规则填充并复制边缘），
    // 所以必须换算成本地行号再取址。曾经这里直接用整图行号索引缓冲，
    // 单 band 时整幅图被平移 halo 行、多 band 时每段都错位，MSE 完全不对。
    const int ly = gy - top;
    const size_t offset =
        (static_cast<size_t>(ly) * width + static_cast<size_t>(x)) * channels +
        c;
    const double diff =
        static_cast<double>(a[offset]) - static_cast<double>(b[offset]);

    local_sq += diff * diff;

    if (with_window != 0) {
      double px_ssim = 0.0;
      double px_cs = 0.0;
      accumulateWindow(a, b, width, height, channels, gy, x, c, top, halo,
                       weights, &px_ssim, &px_cs);
      local_ssim += px_ssim;
      local_cs += px_cs;
    }
  }

  // block 内归约（三份 double）。线程数固定 256（见 metrics_launch.cu）。
  __shared__ double s_ssim[256];
  __shared__ double s_cs[256];
  __shared__ double s_sq[256];

  const int lane = threadIdx.x;
  s_ssim[lane] = local_ssim;
  s_cs[lane] = local_cs;
  s_sq[lane] = local_sq;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      s_ssim[lane] += s_ssim[lane + stride];
      s_cs[lane] += s_cs[lane + stride];
      s_sq[lane] += s_sq[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    atomicAdd(acc_ssim, s_ssim[0]);
    atomicAdd(acc_cs, s_cs[0]);
    atomicAdd(acc_sq, s_sq[0]);
  }
}

} // namespace metrics
} // namespace cuda
} // namespace image_codec
