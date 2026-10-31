/*
 * File: metrics_common.h
 * Brief: Metrics common.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics_common.h - MSE/PSNR/SSIM/MS-SSIM 的共享定义（CPU 参考与 GPU kernel 共用）。
//
// 为什么把这些常数和公式单独抽出来：GPU kernel 与 CPU 参考实现必须给出**同一个
// 数字**，否则「GPU 版画质更好/更差」这种结论根本无从判断。只要有一处常数、
// 边界策略或金字塔构建方式不同，两边的差距就会稳定在 1e-3 量级——肉眼看不出来，
// 但横向比较全部作废。常数、权重和合成公式因此各只有一份：
//   * 与设备无关的部分（权重表、PSNR 换算、MS-SSIM 合成、2x2 下采样）在本文件；
//   * 只出现在设备代码里的部分（11x11 窗求和顺序）在 kernels/metrics.cu。
#pragma once

#include <cstdint>

#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace metrics {

// ---- SSIM 常数（Wang 等 2004，L = 255）----
// 11x11 高斯窗，σ = 1.5，复制边缘（replicate）边界。
constexpr int kWindow = 11;
constexpr int kHalo = kWindow / 2;
constexpr double kSigma = 1.5;
constexpr double kC1 = (0.01 * 255.0) * (0.01 * 255.0);
constexpr double kC2 = (0.03 * 255.0) * (0.03 * 255.0);

// ---- MS-SSIM 尺度权重（Wang 等 2003）----
// 5 个尺度，从原图开始每级 2x2 均值下采样。图像太小时只用前 M 级，
// 并把前 M 个权重**重新归一化**（和为 1），否则小图的 MS-SSIM 会系统性偏小。
constexpr int kMsSsimMaxLevels = 5;
constexpr double kMsSsimWeights[kMsSsimMaxLevels] = {0.0448, 0.2856, 0.3001,
                                                     0.2363, 0.1333};

// 单个尺度的累加量。
//   sq_sum   = Σ(x-y)²      （MSE 用，只有第 0 级有意义）
//   ssim_sum = Σ SSIM(px)   （每 (像素, 通道) 一个 SSIM）
//   cs_sum   = Σ CS(px)     （SSIM 去掉亮度项；MS-SSIM 的中间尺度只用它）
struct ScaleStats {
  double sq_sum = 0.0;
  double ssim_sum = 0.0;
  double cs_sum = 0.0;
  uint64_t samples = 0; // 参与统计的样本数（像素 × 通道）
};

// 归一化后的 11 点一维高斯权重（二维窗 = 外积，见 kernel 与参考实现）。
void gaussianWeights11(double *out);

// MSE -> PSNR。MSE == 0（逐位一致）固定返回 99.0：有限值，避免 Inf/NaN
// 污染平均值与 JSON 输出。
double psnrFromMse(double mse);

// MS-SSIM 实际可用的尺度数：每一级仍需 >= 11x11 才能算 11x11 窗。
int msSsimLevels(int width, int height);

// 2x2 均值下采样（各通道独立）。输出尺寸为 floor(w/2) x floor(h/2)，
// 奇数尾行/尾列被丢弃——与 pytorch-msssim 的 avg_pool2d(kernel=2) 一致，
// 不是 imresize 的双三次（那会引入负值和振铃，让两个实现的数字对不上）。
bool downsample2x(const ImageBuffer &src, ImageBuffer *dst);

// 由各尺度统计合成 MS-SSIM。
//   cs[j]     : 第 j 级（0 基）的 CS 均值，共 levels 个
//   ssim_last : 最末一级的完整 SSIM 均值
// 公式：Π_{j<levels-1} cs[j]^w[j] · ssim_last^w[levels-1]（w 已归一化）
double msSsimFromScales(const double *cs, double ssim_last, int levels);

} // namespace metrics
} // namespace cuda
} // namespace image_codec
