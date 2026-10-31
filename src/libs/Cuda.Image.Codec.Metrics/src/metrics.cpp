/*
 * File: metrics.cpp
 * Brief: Metrics.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics.cpp - MSE / PSNR / SSIM / MS-SSIM 的 CUDA 实现（按行 band 流式上传）。
//
// 为什么按 band 而不是整幅上传：Merge.bmp 是 32000x19656x3（1.89 GB），
// 两张图同时驻留设备就是 3.8 GB，再加 SSIM 中间量必然 OOM。按 band 上传后
// 显存占用只与 band_rows 有关、与图像高度无关，大图与小图走同一条代码路径。
#include "metrics.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#include "cuda_buffer.h"
#include "cuda_device.h"
#include "cuda_error.h"
#include "metrics_common.h"
#include "metrics_launch.h"

namespace image_codec {
namespace cuda {
namespace metrics {

namespace {

// 逐 band 复用的设备缓冲（5 个尺度共用一套，靠 ensure() 只增长不缩小）。
struct Scratch {
  DeviceBuffer a;
  DeviceBuffer b;
  DeviceBuffer acc_ssim;
  DeviceBuffer acc_cs;
  DeviceBuffer acc_sq;

  void bindBudget(MemoryBudget *budget) {
    a.setBudget(budget);
    b.setBudget(budget);
    acc_ssim.setBudget(budget);
    acc_cs.setBudget(budget);
    acc_sq.setBudget(budget);
  }
};

// 参数校验：尺寸/通道必须一致，行宽要能被 kernel 的 int 索引装下。
bool validate(const ImageBuffer &a, const ImageBuffer &b, std::string *err) {
  if (!a.valid() || !b.valid()) {
    if (err != nullptr) *err = "metrics: invalid image buffer";
    return false;
  }
  if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
    if (err != nullptr) *err = "metrics: size/channel mismatch";
    return false;
  }
  if (static_cast<long long>(a.width) * a.channels > INT_MAX / 2) {
    if (err != nullptr) *err = "metrics: image row too wide for the kernel";
    return false;
  }
  return true;
}

// 单尺度统计（GPU）。with_window == 0 时只累加平方误差。
bool scaleStats(const ImageBuffer &a, const ImageBuffer &b, Device &dev,
                Scratch &scratch, int band_rows, int with_window,
                ScaleStats *out, std::string *err) {
  *out = ScaleStats{};

  const int width = a.width;
  const int height = a.height;
  const int channels = a.channels;

  const size_t row_bytes = a.stride();
  const int step = std::clamp(band_rows, 1, height);
  const int rows = step + 2 * kHalo;
  const size_t band_bytes = static_cast<size_t>(rows) * row_bytes;

  std::string local_err;
  if (!scratch.a.ensure(band_bytes, &local_err) ||
      !scratch.b.ensure(band_bytes, &local_err) ||
      !scratch.acc_ssim.ensure(sizeof(double), &local_err) ||
      !scratch.acc_cs.ensure(sizeof(double), &local_err) ||
      !scratch.acc_sq.ensure(sizeof(double), &local_err)) {
    if (err != nullptr) *err = "metrics: " + local_err;
    return false;
  }
  if (!check(cudaMemsetAsync(scratch.acc_ssim.data(), 0, sizeof(double),
                             dev.stream()),
             "cudaMemsetAsync(ssim)", &local_err) ||
      !check(cudaMemsetAsync(scratch.acc_cs.data(), 0, sizeof(double),
                             dev.stream()),
             "cudaMemsetAsync(cs)", &local_err) ||
      !check(cudaMemsetAsync(scratch.acc_sq.data(), 0, sizeof(double),
                             dev.stream()),
             "cudaMemsetAsync(sq)", &local_err)) {
    if (err != nullptr) *err = "metrics: " + local_err;
    return false;
  }

  double weights[kWindow];
  gaussianWeights11(weights);

  // 复用同一块 host 缓冲：每 band 只填 (step + 2*halo) 行，越界行复制边缘行。
  // halo 行只在 with_window 时有意义，但两种路径共用同一次上传更简单，
  // 且 MSE 路径多传的 10 行相对整幅图可以忽略。
  std::vector<uint8_t> host_a(band_bytes);
  std::vector<uint8_t> host_b(band_bytes);

  for (int y0 = 0; y0 < height; y0 += step) {
    const int expect = std::min(step, height - y0);

    for (int r = 0; r < rows; ++r) {
      int src_row = y0 - kHalo + r;
      if (src_row < 0) src_row = 0;
      if (src_row >= height) src_row = height - 1;

      std::memcpy(host_a.data() + static_cast<size_t>(r) * row_bytes,
                  a.data.data() + static_cast<size_t>(src_row) * row_bytes,
                  row_bytes);
      std::memcpy(host_b.data() + static_cast<size_t>(r) * row_bytes,
                  b.data.data() + static_cast<size_t>(src_row) * row_bytes,
                  row_bytes);
    }

    if (!scratch.a.upload(host_a.data(), band_bytes, &local_err,
                          dev.stream()) ||
        !scratch.b.upload(host_b.data(), band_bytes, &local_err,
                          dev.stream())) {
      if (err != nullptr) *err = "metrics: " + local_err;
      return false;
    }

    launchMetricsBand(scratch.a.as<uint8_t>(), scratch.b.as<uint8_t>(), width,
                      height, channels, y0, expect, kHalo, rows, with_window,
                      weights, scratch.acc_ssim.as<double>(),
                      scratch.acc_cs.as<double>(), scratch.acc_sq.as<double>(),
                      dev.stream());

    if (!check(cudaGetLastError(), "metrics_band_kernel", &local_err) ||
        !dev.synchronize(&local_err)) {
      if (err != nullptr) *err = "metrics: " + local_err;
      return false;
    }
  }

  if (!scratch.acc_ssim.download(&out->ssim_sum, sizeof(double), &local_err,
                                 dev.stream()) ||
      !scratch.acc_cs.download(&out->cs_sum, sizeof(double), &local_err,
                               dev.stream()) ||
      !scratch.acc_sq.download(&out->sq_sum, sizeof(double), &local_err,
                               dev.stream())) {
    if (err != nullptr) *err = "metrics: " + local_err;
    return false;
  }

  out->samples = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                 static_cast<uint64_t>(channels);
  return true;
}

} // namespace

bool mse(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
         double *out, std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "metrics: null output";
    return false;
  }
  *out = 0.0;

  if (!validate(a, b, err)) return false;
  if (!options.use_gpu) return reference::mse(a, b, out, err);

  Device dev;
  std::string local_err;
  if (!dev.init(options.device_index, &local_err)) {
    if (err != nullptr) *err = "metrics: " + local_err;
    return false;
  }
  MemoryBudget local_budget;
  MemoryBudget &budget =
      options.budget != nullptr ? *options.budget : local_budget;
  budget.bindDevice(options.device_index);
  budget.setLimit(options.memory_limit);

  Scratch scratch;
  scratch.bindBudget(&budget);

  ScaleStats s;
  if (!scaleStats(a, b, dev, scratch, options.band_rows, /*with_window=*/0, &s,
                  err)) {
    return false;
  }
  const double n = static_cast<double>(s.samples);
  if (!(n > 0.0)) {
    if (err != nullptr) *err = "metrics: empty image";
    return false;
  }
  *out = s.sq_sum / n;
  return true;
}

bool psnr(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
          double *out, std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "metrics: null output";
    return false;
  }
  double m = 0.0;
  if (!mse(a, b, options, &m, err)) return false;
  *out = psnrFromMse(m);
  return true;
}

bool ssim(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
          SsimResult *out, std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "metrics: null output";
    return false;
  }
  *out = SsimResult{};

  if (!validate(a, b, err)) return false;
  if (!options.use_gpu) return reference::ssim(a, b, out, err);

  Device dev;
  std::string local_err;
  if (!dev.init(options.device_index, &local_err)) {
    if (err != nullptr) *err = "metrics: " + local_err;
    return false;
  }
  MemoryBudget local_budget;
  MemoryBudget &budget =
      options.budget != nullptr ? *options.budget : local_budget;
  budget.bindDevice(options.device_index);
  budget.setLimit(options.memory_limit);

  Scratch scratch;
  scratch.bindBudget(&budget);

  // ---- 第 0 级：SSIM 本身 + MS-SSIM 第 0 级的 CS（同一次 11x11 窗统计）----
  ScaleStats s;
  if (!scaleStats(a, b, dev, scratch, options.band_rows, /*with_window=*/1, &s,
                  err)) {
    return false;
  }
  const double n0 = static_cast<double>(s.samples);
  if (!(n0 > 0.0)) {
    if (err != nullptr) *err = "metrics: empty image";
    return false;
  }
  out->ssim = s.ssim_sum / n0;

  // ---- MS-SSIM：逐级 2x2 均值下采样，只有最末一级取完整 SSIM ----
  const int levels = msSsimLevels(a.width, a.height);
  out->levels = levels;

  double cs[kMsSsimMaxLevels] = {0.0, 0.0, 0.0, 0.0, 0.0};
  cs[0] = s.cs_sum / n0;

  double last_ssim = out->ssim;

  // 层级间只保留当前层与下一层：整座金字塔在 32000x19656 上要白占约 630 MB。
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
    if (!scaleStats(next_a, next_b, dev, scratch, options.band_rows,
                    /*with_window=*/1, &sl, err)) {
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

} // namespace metrics
} // namespace cuda
} // namespace image_codec
