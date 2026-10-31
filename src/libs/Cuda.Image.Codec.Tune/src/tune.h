/*
 * File: tune.h
 * Brief: Tune.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tune.h - 参数扫描（Tune）库：在自己的硬件上找到最优编码参数。
//
// 为什么单独做一个库而不是把循环塞进 CLI：GPU 参数（tile 大小、每批单元数、
// block 线程数、PNG 滤波器策略、zlib 等级）的最优点与具体 GPU、显存容量、
// 图像尺寸强相关，别人测出来的数字在你机器上往往不成立。这个库把「跑一遍
// 所有组合、计时、校验、排序」做成可复用的 C++ 接口，CLI 只是它的一层薄壳，
// 上层程序（ImageViewer 的基准页、CI 的性能门限）可以直接链接它。
//
// 约定：
//   * 每个候选都做**往返校验**（TIFF/PNG 逐位一致，JPEG 只校验尺寸+PSNR 下限），
//     因为「快但错」的参数组合没有意义；
//   * 计时取多次迭代的**最小值**：GPU 上有偶发调度抖动，平均值会被一次抖动污染；
//   * 测量走的是与 C API 完全相同的模块入口，只有一处差别：不复制输入图
//     （C API 为了稳定 ABI 会复制一份 host 缓冲），因此这里测的是编码本身。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda_memory.h"
#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace tune {

enum class Format { Tiff = 0, Png = 1, Jpeg = 2 };

const char *formatName(Format format);

struct Options {
  int device_index = 0;
  int iterations = 3;         // 每个候选的计时次数（取最小值）
  int warmup = 1;             // 预热次数（不计时，用于建立 context / 缓存）
  size_t memory_limit = 0;    // 显存上限（0 = 不限制，仅受设备空闲显存约束）
  bool verify = true;         // 往返校验
  bool stop_on_error = false; // 某个候选失败后是否继续
};

// 一个候选参数组合。字段按格式取用，未用到的忽略。
struct Candidate {
  std::string label;
  // TIFF
  int tiff_tile_size = 0;
  int tiff_rows_per_strip = 0;
  int tiff_compression = 5;
  int tiff_lzw_block_threads = 0;
  int tiff_lzw_unit_batch = 0;
  // PNG
  int png_filter = -1;
  int png_compression = 6;
  int png_block_threads = 0;
  // JPEG
  int jpeg_quality = 90;
  int jpeg_subsampling = 0;
};

struct Result {
  std::string label;
  bool ok = false;
  std::string error;
  double best_ms = 0.0;        // 最优单次耗时
  double median_ms = 0.0;      // 中位数（看稳定性）
  double input_mb_per_s = 0.0; // 输入吞吐（MB/s，按 1e6 字节）
  size_t output_bytes = 0;
  size_t peak_vram = 0; // 本次候选的显存峰值
  bool gpu_used = false;
  double psnr = 0.0; // 仅 JPEG（有损）有意义
  double ssim = 0.0;
  double ms_ssim = 0.0;
};

// ---- 默认候选集 ----
// TIFF: tile ∈ {0(strip),128,256,512,1024} × rows ∈ {8,32,128} × threads ∈ {128,256}
// PNG : filter ∈ {-1,0,1,2,3,4} × level ∈ {1,6,9} × threads ∈ {128,256,512}
// JPEG: quality ∈ {90,95} × subsampling ∈ {444,420,422}
std::vector<Candidate> defaultCandidates(Format format);

// ---- 单候选择时 ----
// 注意：verify 失败时 ok=false，best_ms 仍会被填上（便于看「错的那条有多快」）。
Result run(const ImageBuffer &img, Format format, const Candidate &candidate,
           const Options &options);

// ---- 整组扫描（按 best_ms 升序，失败项排最后）----
std::vector<Result> sweep(const ImageBuffer &img, Format format,
                          const std::vector<Candidate> &candidates,
                          const Options &options);

// ---- 报告 ----
std::string toMarkdown(const std::vector<Result> &results, Format format);
std::string toJson(const std::vector<Result> &results, Format format);

// ---- 读图（调参用）----
// 直接从 24-bit BMP 里按需读取一个矩形区域，**不**把整幅图读进内存：
// 本机 Merge.bmp 是 32000x19656（1.89 GB），整幅读入再裁剪纯属浪费。
// crop_w/crop_h <= 0 表示读到图像边界为止。x/y 以图像左上角为原点。
bool loadBmpRegion(const char *path, int x, int y, int crop_w, int crop_h,
                   ImageBuffer *out, std::string *err);

} // namespace tune
} // namespace cuda
} // namespace image_codec
