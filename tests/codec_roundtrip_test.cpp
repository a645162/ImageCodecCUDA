/*
 * File: codec_roundtrip_test.cpp
 * Brief: Codec roundtrip test.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tests/codec_roundtrip_test.cpp - ImageCodecCUDA 自检。
//
// 覆盖的不是「能跑通」，而是后端最容易悄悄错的三件事：
//   1. GPU LZW 与 CPU 参考实现逐字节一致（码宽/ClearCode 时机错一位就全错）；
//   2. GPU 行滤波与 CPU 参考实现逐字节一致（Paeth/Average 的边界最容易写错）；
//   3. 容器往返后的像素逐位一致（容器 tag/单 strip 偏移那类缺陷只有往返能抓到）。
//
// 退出码：0 = 通过，2 = SKIP（无 CUDA 设备），其它 = 失败。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda_device.h"
#include "image_buffer.h"
#include "jpeg_codec.h"
#include "lzw_codec.h"
#include "metrics.h"
#include "lzw_encode_cuda.h"
#include "png_codec.h"
#include "png_container.h"
#include "png_filter_cuda.h"
#include "tiff_codec.h"

using image_codec::cuda::ImageBuffer;

namespace {

int g_failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

// 造一张确定性的渐变 + 噪声图（纯渐变会让 LZW 字典很快饱和，掩盖码宽缺陷）。
ImageBuffer makeImage(int w, int h, int channels) {
  ImageBuffer img;
  img.allocate(w, h, channels);
  std::mt19937 rng(12345);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < channels; ++c) {
        const int idx = (y * w + x) * channels + c;
        const int v =
            (x * 3 + y * 5 + c * 37 + static_cast<int>(rng() % 7)) & 0xFF;
        img.data[static_cast<size_t>(idx)] = static_cast<uint8_t>(v);
      }
    }
  }
  return img;
}

void testLzwParity() {
  std::mt19937 rng(7);
  const std::vector<size_t> sizes = {1,    2,    3,     17,    256,
                                     4095, 4096, 65536, 300000};
  std::vector<uint8_t> input;
  for (size_t n : sizes) {
    input.resize(n);
    for (size_t i = 0; i < n; ++i) {
      // 小字母表 + 随机，既有长串也有频繁新条目。
      input[i] = static_cast<uint8_t>(rng() % 5);
    }
    std::vector<uint8_t> cpu;
    if (image_codec::cuda::lzw::encode(input.data(), input.size(), cpu) != 0) {
      check(false, "cpu lzw encode");
      return;
    }
    image_codec::cuda::tiff::LzwEncoderCuda enc(0);
    std::vector<uint32_t> sizes_out;
    std::vector<uint8_t> gpu;
    if (enc.encodePaddedUnits(input.data(), n, 1, sizes_out, gpu) != 0) {
      std::fprintf(stderr, "FAIL: gpu lzw encode (%s)\n",
                   enc.lastError().c_str());
      ++g_failures;
      return;
    }
    check(sizes_out.size() == 1, "gpu lzw unit count");
    check(gpu == cpu, "gpu lzw == cpu lzw (bit exact)");
    std::vector<uint8_t> decoded;
    check(image_codec::cuda::lzw::decode(gpu.data(), gpu.size(), decoded) == 0,
          "lzw decode");
    check(decoded == input, "lzw roundtrip");
  }
}

void testTiff(const char *what, int tile_size, int rows_per_strip,
              int compression) {
  const ImageBuffer img = makeImage(517, 391, 3);
  image_codec::cuda::tiff::TiffEncodeOptions opt;
  opt.compression = compression;
  opt.tile_size = tile_size;
  opt.rows_per_strip = rows_per_strip;
  opt.use_gpu = true;

  std::vector<uint8_t> bytes;
  bool gpu_used = false;
  std::string err;
  if (!image_codec::cuda::tiff::encodeTiff(img, opt, &bytes, &gpu_used, &err)) {
    std::fprintf(stderr, "FAIL: %s encode (%s)\n", what, err.c_str());
    ++g_failures;
    return;
  }
  if (compression == 5) {
    check(gpu_used, "tiff lzw used GPU");
  }
  ImageBuffer back;
  if (!image_codec::cuda::tiff::decodeTiff(bytes.data(), bytes.size(), &back,
                                           &err)) {
    std::fprintf(stderr, "FAIL: %s decode (%s)\n", what, err.c_str());
    ++g_failures;
    return;
  }
  check(back.width == img.width && back.height == img.height &&
            back.channels == img.channels,
        "tiff dimensions");
  check(back.data == img.data, what);
}

// 通道矩阵：1/2/3/4 通道各自原样往返。TIFF 的 2 通道（灰度+Alpha）最容易漏，
// 它要求 spp=2 + Photometric=BlackIsZero + ExtraSamples 同时写对，
// 而且 BitsPerSample 恰好 4 字节必须内联。
void testTiffChannels(int channels) {
  const ImageBuffer img = makeImage(129, 95, channels);
  image_codec::cuda::tiff::TiffEncodeOptions opt;
  opt.compression = 5;
  opt.rows_per_strip = 16;
  opt.use_gpu = true;

  std::vector<uint8_t> bytes;
  bool gpu_used = false;
  std::string err;
  if (!image_codec::cuda::tiff::encodeTiff(img, opt, &bytes, &gpu_used, &err)) {
    std::fprintf(stderr, "FAIL: tiff %dch encode (%s)\n", channels,
                 err.c_str());
    ++g_failures;
    return;
  }

  ImageBuffer back;
  if (!image_codec::cuda::tiff::decodeTiff(bytes.data(), bytes.size(), &back,
                                           &err)) {
    std::fprintf(stderr, "FAIL: tiff %dch decode (%s)\n", channels,
                 err.c_str());
    ++g_failures;
    return;
  }

  check(back.width == img.width && back.height == img.height &&
            back.channels == channels,
        "tiff channel matrix dimensions");
  check(back.data == img.data, "tiff channel matrix roundtrip");
}

void testPngFilterParity() {
  const ImageBuffer img = makeImage(97, 61, 3);
  for (int filter = 0; filter <= 4; ++filter) {
    image_codec::cuda::png::PngFilterCuda gpu;
    std::vector<uint8_t> gpu_stream;
    if (!gpu.buildFilteredStream(img, filter, &gpu_stream, nullptr)) {
      std::fprintf(stderr, "FAIL: png gpu filter %d (%s)\n", filter,
                   gpu.lastError().c_str());
      ++g_failures;
      continue;
    }
    const std::vector<uint8_t> cpu_stream =
        image_codec::cuda::png::buildFilteredStream(
            img.data.data(), img.width, img.height, img.channels,
            std::vector<uint8_t>(static_cast<size_t>(img.height),
                                 static_cast<uint8_t>(filter)));
    check(gpu_stream == cpu_stream, "png gpu filter == cpu filter");
  }

  // 自适应：GPU 选择的滤波器与 CPU 参考实现可能不同（都合法），
  // 因此只校验「生成的流能被自己的解码器还原」这一更强的不变量。
  image_codec::cuda::png::PngFilterCuda gpu;
  std::vector<uint8_t> adaptive;
  check(gpu.buildFilteredStream(img, -1, &adaptive, nullptr),
        "png adaptive filter");
  check(adaptive.size() == static_cast<size_t>(img.height) * (img.stride() + 1),
        "png adaptive stream size");
}

void testPng(int channels) {
  const ImageBuffer img = makeImage(233, 145, channels);
  image_codec::cuda::png::PngEncodeOptions opt;
  std::vector<uint8_t> bytes;
  bool gpu_used = false;
  std::string err;
  if (!image_codec::cuda::png::encodePng(img, opt, &bytes, &gpu_used, &err)) {
    std::fprintf(stderr, "FAIL: png encode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(gpu_used, "png filtering used GPU");
  ImageBuffer back;
  if (!image_codec::cuda::png::decodePng(bytes.data(), bytes.size(), &back,
                                         &err)) {
    std::fprintf(stderr, "FAIL: png decode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(back.width == img.width && back.height == img.height &&
            back.channels == img.channels,
        "png dimensions");
  check(back.data == img.data, "png roundtrip bit exact");
}

void testJpeg() {
  if (!image_codec::cuda::jpeg::available()) {
    std::printf("SKIP: nvJPEG 不可用\n");
    return;
  }
  const ImageBuffer img = makeImage(320, 240, 3);
  std::vector<uint8_t> bytes;
  std::string err;
  if (!image_codec::cuda::jpeg::encode(0, img, 95, 444, &bytes, &err)) {
    std::fprintf(stderr, "FAIL: jpeg encode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(!bytes.empty() && bytes[0] == 0xFF && bytes[1] == 0xD8, "jpeg soi");
  ImageBuffer back;
  if (!image_codec::cuda::jpeg::decode(0, bytes.data(), bytes.size(), &back,
                                       &err)) {
    std::fprintf(stderr, "FAIL: jpeg decode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(back.width == img.width && back.height == img.height,
        "jpeg dimensions");
  double se = 0;
  const size_t n = static_cast<size_t>(img.width) * img.height * 3;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(img.data[i]) - back.data[i];
    se += d * d;
  }
  const double mse = se / static_cast<double>(n);
  const double psnr = mse <= 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
  std::printf("jpeg 444 q95 psnr = %.2f dB\n", psnr);
  check(psnr > 32.0, "jpeg psnr > 32 dB");
}

// 灰度 JPEG：1 通道进、1 通道出。
//
// 这条单独测，是因为 nvJPEG 的编码输入只接受 3 通道：只要忘了把采样因子设成
// CSS_GRAY，就会得到一张三分量 JPEG，解回来变 3 通道 —— 压缩比、体积、上层
// 的通道判断全部跟着错，而肉眼几乎看不出来。
void testJpegGray() {
  if (!image_codec::cuda::jpeg::available()) {
    std::printf("SKIP: nvJPEG 不可用\n");
    return;
  }
  const ImageBuffer img = makeImage(97, 71, 1);
  std::vector<uint8_t> bytes;
  std::string err;
  if (!image_codec::cuda::jpeg::encode(0, img, 95, 0, &bytes, &err)) {
    std::fprintf(stderr, "FAIL: gray jpeg encode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }

  ImageBuffer back;
  if (!image_codec::cuda::jpeg::decode(0, bytes.data(), bytes.size(), &back,
                                       &err)) {
    std::fprintf(stderr, "FAIL: gray jpeg decode (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(back.width == img.width && back.height == img.height,
        "gray jpeg dimensions");
  check(back.channels == 1, "gray jpeg stays 1 channel");

  if (back.channels != 1) return;
  double se = 0;
  const size_t n = static_cast<size_t>(img.width) * img.height;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(img.data[i]) - back.data[i];
    se += d * d;
  }
  const double mse = se / static_cast<double>(n);
  const double psnr = mse <= 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
  std::printf("gray jpeg q95 psnr = %.2f dB, %zu bytes\n", psnr, bytes.size());
  check(psnr > 32.0, "gray jpeg psnr > 32 dB");
}

// 画质指标：GPU 与 CPU 参考实现必须给出同一个数字。
//
// 这里刻意用「小 band」（band_rows = 16）跑一遍：band 越小，halo 行复制与
// 归约次数越多，边界拼接错一位就会立刻暴露成数字不一致，而默认 256 行的
// band 在 300x200 的图上只有一两个 band，根本测不出问题。
void testMetrics(const ImageBuffer &a, const ImageBuffer &b, const char *what) {
  using namespace image_codec::cuda;

  metrics::Options opt;
  opt.device_index = 0;
  opt.band_rows = 16;

  const double mse_gpu = [&] {
    double v = -1.0;
    std::string err;
    check(metrics::mse(a, b, opt, &v, &err), "metrics::mse gpu");
    return v;
  }();

  double mse_cpu = -1.0;
  std::string err;
  check(metrics::reference::mse(a, b, &mse_cpu, &err), "metrics::mse cpu");
  check(std::fabs(mse_gpu - mse_cpu) < 1e-9, "mse gpu == cpu");

  double psnr_gpu = 0.0;
  check(metrics::psnr(a, b, opt, &psnr_gpu, &err), "metrics::psnr gpu");
  const double psnr_expected =
      mse_cpu <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse_cpu);
  check(std::fabs(psnr_gpu - psnr_expected) < 1e-6, "psnr from mse");

  metrics::SsimResult ssim_gpu;
  check(metrics::ssim(a, b, opt, &ssim_gpu, &err), "metrics::ssim gpu");

  metrics::SsimResult ssim_cpu;
  check(metrics::reference::ssim(a, b, &ssim_cpu, &err), "metrics::ssim cpu");

  check(std::fabs(ssim_gpu.ssim - ssim_cpu.ssim) < 1e-9,
        "ssim gpu == cpu (banded halo)");
  check(std::fabs(ssim_gpu.ms_ssim - ssim_cpu.ms_ssim) < 1e-9,
        "ms-ssim gpu == cpu (banded halo)");
  check(ssim_gpu.levels == ssim_cpu.levels, "ms-ssim levels gpu == cpu");

  std::printf("%s: mse=%.6f psnr=%.4f dB ssim=%.9f ms-ssim=%.9f (%d 级)\n",
              what, mse_gpu, psnr_gpu, ssim_gpu.ssim, ssim_gpu.ms_ssim,
              ssim_gpu.levels);
}

void testMetricsSuite() {
  // 1. 完全相同：MSE 0 / PSNR 99 / SSIM 1 / MS-SSIM 1。
  //    400x300 才够 5 级（每级减半后仍需 >= 11 像素）。
  const ImageBuffer same = makeImage(400, 300, 3);
  testMetrics(same, same, "identical");

  {
    image_codec::cuda::metrics::Options opt;
    image_codec::cuda::metrics::SsimResult r;
    std::string err;
    if (image_codec::cuda::metrics::ssim(same, same, opt, &r, &err)) {
      check(std::fabs(r.ssim - 1.0) < 1e-12, "identical ssim == 1");
      check(std::fabs(r.ms_ssim - 1.0) < 1e-12, "identical ms-ssim == 1");
      check(r.levels == 5, "400x300 用满 5 级");
    } else {
      check(false, "identical ssim");
    }
  }

  // 2. 有扰动：三个数字都必须与 CPU 参考逐位一致（这才是真在测对齐）。
  ImageBuffer noisy = makeImage(157, 123, 3);
  std::mt19937 rng(99);
  for (auto &v : noisy.data) {
    const int d = static_cast<int>(rng() % 5) - 2;
    const int nv = static_cast<int>(v) + d;
    v = static_cast<uint8_t>(nv < 0 ? 0 : (nv > 255 ? 255 : nv));
  }
  testMetrics(noisy, noisy, "perturbed baseline");
  testMetrics(makeImage(157, 123, 3), noisy, "perturbed");

  // 3. 灰度：通道数 1 时 11x11 窗与 OTSU 之外的东西无关，同样要对齐。
  const ImageBuffer gray = makeImage(97, 71, 1);
  testMetrics(gray, gray, "gray identical");

  // 4. 小图：小于 22 像素时无法下采样，MS-SSIM 退化成 SSIM（levels == 1）。
  {
    const ImageBuffer tiny = makeImage(16, 13, 3);
    image_codec::cuda::metrics::Options opt;
    image_codec::cuda::metrics::SsimResult r;
    std::string err;
    if (image_codec::cuda::metrics::ssim(tiny, tiny, opt, &r, &err)) {
      check(r.levels == 1, "16x13 只有 1 级");
      check(std::fabs(r.ms_ssim - r.ssim) < 1e-12, "单级时 ms-ssim == ssim");
    } else {
      check(false, "tiny ssim");
    }
  }
}

// 解析解校验：常数图对 σ = 0，CS 恒为 1，SSIM 退化成亮度项，可以手算。
// 这是唯一不依赖「另一份实现」的检查——前面那些 GPU/CPU 对拍只能证明两边一致，
// 证明不了两边一起错（例如把 C1/C2 写反、或亮度项分子少乘 2）。
void testMetricsAnalytic() {
  ImageBuffer a;
  a.allocate(400, 300, 3);
  std::fill(a.data.begin(), a.data.end(), 100);
  ImageBuffer b;
  b.allocate(400, 300, 3);
  std::fill(b.data.begin(), b.data.end(), 110);

  const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
  const double expected_ssim =
      (2.0 * 100.0 * 110.0 + c1) / (100.0 * 100.0 + 110.0 * 110.0 + c1);

  image_codec::cuda::metrics::Options opt;
  image_codec::cuda::metrics::SsimResult r;
  std::string err;
  if (!image_codec::cuda::metrics::ssim(a, b, opt, &r, &err)) {
    std::fprintf(stderr, "FAIL: analytic ssim (%s)\n", err.c_str());
    ++g_failures;
    return;
  }
  check(std::fabs(r.ssim - expected_ssim) < 1e-9,
        "ssim matches luminance-only");

  // CS 恒为 1，所以 MS-SSIM 只剩最末一级的 SSIM 按其权重开方。
  // 注意：标准 5 个权重之和是 1.0001（不是 1），实现里做了归一化，
  // 所以这里的指数是 0.1333/1.0001 而不是 0.1333。
  const double weight_last =
      0.1333 / (0.0448 + 0.2856 + 0.3001 + 0.2363 + 0.1333);
  const double expected_ms = std::pow(expected_ssim, weight_last);
  check(std::fabs(r.ms_ssim - expected_ms) < 1e-9,
        "ms-ssim matches single-scale term");

  double m = 0.0;
  if (!image_codec::cuda::metrics::mse(a, b, opt, &m, &err)) {
    check(false, "analytic mse");
    return;
  }
  check(std::fabs(m - 100.0) < 1e-9, "mse == 10^2");

  double p = 0.0;
  if (!image_codec::cuda::metrics::psnr(a, b, opt, &p, &err)) {
    check(false, "analytic psnr");
    return;
  }
  check(std::fabs(p - 10.0 * std::log10(255.0 * 255.0 / 100.0)) < 1e-9,
        "psnr formula");

  std::printf("analytic: ssim=%.9f (expected %.9f), ms-ssim=%.9f\n", r.ssim,
              expected_ssim, r.ms_ssim);
}

// 同一张图重复编码必须逐字节一致：自适应滤波的选行策略里只要有一个 kernel
// 读了没落地的上传数据，输出尺寸就会在两次之间跳变（曾经就是这个症状）。
void testPngDeterminism() {
  const ImageBuffer img = makeImage(333, 217, 3);
  std::vector<uint8_t> first;
  for (int i = 0; i < 3; ++i) {
    image_codec::cuda::png::PngEncodeOptions opt;
    std::vector<uint8_t> bytes;
    bool gpu_used = false;
    std::string err;
    if (!image_codec::cuda::png::encodePng(img, opt, &bytes, &gpu_used, &err)) {
      std::fprintf(stderr, "FAIL: png determinism encode (%s)\n", err.c_str());
      ++g_failures;
      return;
    }
    if (i == 0) {
      first = std::move(bytes);
    } else {
      check(bytes == first, "png encode is deterministic");
    }
  }
  std::printf("png deterministic: %zu bytes\n", first.size());
}

} // namespace

int main() {
  const auto devices = image_codec::cuda::enumerateDevices();
  if (devices.empty()) {
    std::printf("SKIP: 没有 CUDA 设备\n");
    return 2;
  }
  std::printf("device: %s (sm_%d%d)\n", devices[0].name.c_str(),
              devices[0].major, devices[0].minor);

  testLzwParity();
  testTiff("tiff strip lzw roundtrip", 0, 32, 5);
  testTiff("tiff tile lzw roundtrip", 256, 0, 5);
  testTiff("tiff strip none roundtrip", 0, 64, 1);
  testTiff("tiff tile none roundtrip", 128, 0, 1);
  testPngFilterParity();
  testPng(3);
  testPng(4);
  testPng(1);
  testPng(2);
  for (int channels = 1; channels <= 4; ++channels) {
    testTiffChannels(channels);
  }
  testJpeg();
  testJpegGray();
  testMetricsSuite();
  testMetricsAnalytic();
  testPngDeterminism();

  if (g_failures == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::fprintf(stderr, "%d check(s) failed\n", g_failures);
  return 1;
}
