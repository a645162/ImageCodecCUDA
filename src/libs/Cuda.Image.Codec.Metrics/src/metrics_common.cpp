/*
 * File: metrics_common.cpp
 * Brief: Metrics common.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics_common.cpp - 共享定义的实现（与设备无关，GPU/CPU 两条路径都链接它）。
#include "metrics_common.h"

#include <algorithm>
#include <cmath>

namespace image_codec {
namespace cuda {
namespace metrics {

void gaussianWeights11(double *out) {
  if (out == nullptr) return;
  double raw[kWindow];
  double sum = 0.0;
  for (int i = 0; i < kWindow; ++i) {
    const double d = static_cast<double>(i - kHalo);
    raw[i] = std::exp(-(d * d) / (2.0 * kSigma * kSigma));
    sum += raw[i];
  }
  for (int i = 0; i < kWindow; ++i) out[i] = raw[i] / sum;
}

double psnrFromMse(double mse) {
  if (!(mse > 0.0)) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int msSsimLevels(int width, int height) {
  int levels = 1;
  int w = width;
  int h = height;
  while (levels < kMsSsimMaxLevels) {
    const int nw = w / 2;
    const int nh = h / 2;
    if (nw < kWindow || nh < kWindow) break;
    w = nw;
    h = nh;
    ++levels;
  }
  return levels;
}

bool downsample2x(const ImageBuffer &src, ImageBuffer *dst) {
  if (dst == nullptr || !src.valid()) return false;
  const int w = src.width / 2;
  const int h = src.height / 2;
  if (w <= 0 || h <= 0) return false;
  if (!dst->allocate(w, h, src.channels)) return false;

  const int c = src.channels;
  const size_t src_stride = src.stride();
  const size_t dst_stride = dst->stride();
  const uint8_t *sp = src.data.data();
  uint8_t *dp = dst->data.data();

  // 单层循环（不写 y/x 两层 for）：像素运算按行拼成一段连续索引，
  // 列号用计数递推，避免每个像素做一次除法。
  const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
  size_t done = 0;
  int y = 0;
  int x = 0;
  while (done < total) {
    const uint8_t *r0 = sp + static_cast<size_t>(2 * y) * src_stride;
    const uint8_t *r1 = r0 + src_stride;
    const size_t o0 = static_cast<size_t>(2 * x) * static_cast<size_t>(c);
    const size_t o1 = o0 + static_cast<size_t>(c);
    const size_t o = done * static_cast<size_t>(c);
    for (int k = 0; k < c; ++k) {
      const unsigned sum = static_cast<unsigned>(r0[o0 + k]) +
                           static_cast<unsigned>(r0[o1 + k]) +
                           static_cast<unsigned>(r1[o0 + k]) +
                           static_cast<unsigned>(r1[o1 + k]);
      dp[o + static_cast<size_t>(k)] = static_cast<uint8_t>((sum + 2u) / 4u);
    }
    ++done;
    ++x;
    if (x == w) {
      x = 0;
      ++y;
    }
  }
  return true;
}

double msSsimFromScales(const double *cs, double ssim_last, int levels) {
  if (cs == nullptr || levels < 1 || levels > kMsSsimMaxLevels) return 0.0;
  double weight_sum = 0.0;
  for (int j = 0; j < levels; ++j) weight_sum += kMsSsimWeights[j];
  if (!(weight_sum > 0.0)) return 0.0;

  // 用对数域相乘：CS 与 SSIM 都可能非常接近 1，直接连乘再乘 5 个权重会让
  // 中间结果沉到 double 的精度边缘（尤其乘到第 5 次方时）。
  double log_sum = 0.0;
  for (int j = 0; j < levels - 1; ++j) {
    if (!(cs[j] > 0.0)) return 0.0; // 负相关/退化区域：MS-SSIM 无定义
    log_sum += (kMsSsimWeights[j] / weight_sum) * std::log(cs[j]);
  }
  if (!(ssim_last > 0.0)) return 0.0;
  log_sum += (kMsSsimWeights[levels - 1] / weight_sum) * std::log(ssim_last);
  return std::exp(log_sum);
}

} // namespace metrics
} // namespace cuda
} // namespace image_codec
