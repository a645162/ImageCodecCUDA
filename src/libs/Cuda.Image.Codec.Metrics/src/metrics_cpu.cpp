/*
 * File: metrics_cpu.cpp
 * Brief: Metrics cpu.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics_cpu.cpp - MSE / SSIM / MS-SSIM 的 CPU 参考实现。
//
// 存在的意义不是「CPU 更快」，而是给 GPU kernel 一个可以逐项对拍的基准：
// SSIM 的常数、边界策略、金字塔下采样方式只要有一处不同，两边的数字就会对不上，
// 而这种偏差在真实图像上往往只差 0.001——肉眼与「看起来合理」都发现不了。
//
// 因此这里的写法刻意保持「照着公式直译」：不做 SIMD、不做循环展开、不做累加
// 顺序优化。它跑得慢是设计的一部分，别在这里做性能优化。
#include "metrics.h"

#include <cmath>
#include <vector>

#include "metrics_common.h"

namespace image_codec {
namespace cuda {
namespace metrics {
namespace reference {

namespace {

// 单尺度统计（CPU）。与 kernels/metrics.cu 的 accumulateWindow 逐项对应。
bool scaleStats(const ImageBuffer &a, const ImageBuffer &b, bool with_window,
                ScaleStats *out, std::string *err) {
  if (!a.valid() || !b.valid() || a.width != b.width || a.height != b.height ||
      a.channels != b.channels) {
    if (err != nullptr) *err = "metrics: invalid image pair";
    return false;
  }

  *out = ScaleStats{};

  const int width = a.width;
  const int height = a.height;
  const int channels = a.channels;

  std::vector<double> w(kWindow);
  gaussianWeights11(w.data());

  double sq_sum = 0.0;
  double ssim_sum = 0.0;
  double cs_sum = 0.0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const size_t offset =
            (static_cast<size_t>(y) * width + x) * channels + c;

        const double d = static_cast<double>(a.data[offset]) -
                         static_cast<double>(b.data[offset]);
        sq_sum += d * d;

        if (!with_window) continue;

        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        double sxy = 0.0;

        for (int dy = -kHalo; dy <= kHalo; ++dy) {
          int cy = y + dy;
          if (cy < 0) cy = 0;
          if (cy >= height) cy = height - 1;

          const double wy = w[static_cast<size_t>(dy + kHalo)];

          for (int dx = -kHalo; dx <= kHalo; ++dx) {
            int cx = x + dx;
            if (cx < 0) cx = 0;
            if (cx >= width) cx = width - 1;

            const size_t o =
                (static_cast<size_t>(cy) * width + cx) * channels + c;
            const double px = a.data[o];
            const double py = b.data[o];
            const double g = wy * w[static_cast<size_t>(dx + kHalo)];

            sx += g * px;
            sy += g * py;
            sxx += g * px * px;
            syy += g * py * py;
            sxy += g * px * py;
          }
        }

        const double vx = sxx - sx * sx;
        const double vy = syy - sy * sy;
        const double vxy = sxy - sx * sy;

        const double lum = (2.0 * sx * sy + kC1) / (sx * sx + sy * sy + kC1);
        const double den = vx + vy + kC2;
        const double cs = den > 0.0 ? (2.0 * vxy + kC2) / den : 0.0;

        cs_sum += cs;
        ssim_sum += lum * cs;
      }
    }
  }

  out->sq_sum = sq_sum;
  out->ssim_sum = ssim_sum;
  out->cs_sum = cs_sum;
  out->samples = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                 static_cast<uint64_t>(channels);
  return true;
}

} // namespace

bool mse(const ImageBuffer &a, const ImageBuffer &b, double *out,
         std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "metrics: null output";
    return false;
  }

  ScaleStats s;
  if (!scaleStats(a, b, /*with_window=*/false, &s, err)) return false;

  const double n = static_cast<double>(s.samples);
  if (!(n > 0.0)) {
    if (err != nullptr) *err = "metrics: empty image";
    return false;
  }

  *out = s.sq_sum / n;
  return true;
}

bool ssim(const ImageBuffer &a, const ImageBuffer &b, SsimResult *out,
          std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "metrics: null output";
    return false;
  }
  *out = SsimResult{};

  ScaleStats s;
  if (!scaleStats(a, b, /*with_window=*/true, &s, err)) return false;

  const double n0 = static_cast<double>(s.samples);
  if (!(n0 > 0.0)) {
    if (err != nullptr) *err = "metrics: empty image";
    return false;
  }
  out->ssim = s.ssim_sum / n0;

  const int levels = msSsimLevels(a.width, a.height);
  out->levels = levels;

  double cs[kMsSsimMaxLevels] = {0.0, 0.0, 0.0, 0.0, 0.0};
  cs[0] = s.cs_sum / n0;

  double last_ssim = out->ssim;

  ImageBuffer cur_a;
  ImageBuffer cur_b;
  ImageBuffer next_a;
  ImageBuffer next_b;

  for (int level = 1; level < levels; ++level) {
    const ImageBuffer &src_a = level == 1 ? a : cur_a;
    const ImageBuffer &src_b = level == 1 ? b : cur_b;

    if (!downsample2x(src_a, &next_a) || !downsample2x(src_b, &next_b)) {
      if (err != nullptr) *err = "metrics: pyramid downsampling failed";
      return false;
    }

    ScaleStats sl;
    if (!scaleStats(next_a, next_b, /*with_window=*/true, &sl, err)) {
      return false;
    }

    const double nl = static_cast<double>(sl.samples);
    cs[level] = sl.cs_sum / nl;
    last_ssim = sl.ssim_sum / nl;

    cur_a = std::move(next_a);
    cur_b = std::move(next_b);
  }

  out->ms_ssim = msSsimFromScales(cs, last_ssim, levels);
  return true;
}

} // namespace reference
} // namespace metrics
} // namespace cuda
} // namespace image_codec
