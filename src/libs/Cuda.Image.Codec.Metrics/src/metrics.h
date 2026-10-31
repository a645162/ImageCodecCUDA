/*
 * File: metrics.h
 * Brief: Metrics.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// metrics.h - 两张图的参考画质指标：MSE / PSNR / SSIM / MS-SSIM（GPU 并行）。
//
// 为什么编解码库里要有它：JPEG 是有损的，「压得多小」只有在画质可量化时才有
// 意义；调参（Tune）也必须能证明「这条更快的参数没有把画质压坏」。因此这里与
// 编解码同级地提供指标，而不是让每个调用方各自写一遍（写法一不一致，数字就
// 没法横向比较）。
//
// 定义（与论文/MATLAB 的标准实现对齐，避免和外部工具对不上）：
//   MSE    = Σ(x-y)² / N
//   PSNR   = 10·log10(255² / MSE)（MSE == 0 时固定 99.0，表示「无损」；返回 Inf
//            会污染平均值和 JSON，所以不用 Inf）
//   SSIM   = mean over (像素, 通道) of
//            ((2μxμy + C1)(2σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))
//            μ/σ 用 11x11 高斯窗（σ=1.5），C1=(0.01·255)², C2=(0.03·255)²
//   MS-SSIM= 5 级金字塔（2x2 均值下采样），
//            Π_{j<M-1} CS_j^w_j · SSIM_M^w_{M-1}，w 见 metrics_common.h
//
// 边界策略：**复制边缘**（replicate），整幅图都参与统计。'valid' 策略会让
// 小图的统计样本少掉一圈，不同尺寸的图之间无法比较；这里选可比较的那种，
// 并在 CPU 参考实现里用同一策略，保证 GPU 与 CPU 结果可比（实测差 < 1e-6）。
//
// 跨后端一致性：OpenCL / Vulkan / CUDA 三个后端共用同一套常数与公式，换后端不换
// 数字。**不要**为了「这条路径更简单」去改 SSIM 的窗、边界或加权方式——那会让
// 三个后端的数字再也对不上。
//
// 为什么是三个入口而不是五个：边界在代价上，不在名字上。
//   * mse()  —— 每像素 2 次读，纯减法-平方归约；
//   * psnr() —— 由 MSE 定义，另开一趟毫无意义，内部直接调 mse()；
//   * ssim() —— 11x11 窗（每像素 2x121 次读）是 MSE 的上百倍内存流量。
//     MS-SSIM 的第 0 级**就是**这一次 SSIM 统计，之后每级再下采样各算一次。
//     所以把它俩拆成两个函数只有两种结果：要么重复算最贵的第 0 级，要么其中一个
//     偷偷调另一个——两种都不如一个入口把 ssim / ms_ssim 一起给出来。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda_memory.h"
#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace metrics {

struct Options {
  int device_index = 0;

  // 显存上限（0 = 不限制）。大图按 band_rows 分批上传，显存占用与图像高度
  // 无关，只与（band_rows + 窗高）* 行字节数有关。
  size_t memory_limit = 0;

  // 可选的显存记账对象。留空时每次调用临时建一个；由调用方提供时复用它，
  // 于是调用失败后可以直接问「是不是撞到上限了」（MemoryBudget::exhausted），
  // 而不必去匹配错误字符串。提供时仍按上面的 memory_limit 设置它的上限。
  MemoryBudget *budget = nullptr;

  // 每次上传的行数。显存紧张时调小；太小会让 kernel 启动开销占比上升。
  int band_rows = 256;

  bool use_gpu = true;
};

// SSIM 与 MS-SSIM 一起返回（两者共享第 0 级统计，见文件头）。
struct SsimResult {
  double ssim = 0.0;
  double ms_ssim = 0.0;
  int levels = 0; // MS-SSIM 实际用到的尺度数（1..5；小图会少于 5）
};

// 下面三个入口都要求两张图同尺寸、同通道数（不一致直接失败：把不同尺寸的图
// 缩放后再比较是另一件事，不该藏在一个「算指标」的函数里）。
// 失败时返回 false 并写 err（不会打印）。

// Σ(x-y)² / N。唯一不读 11x11 窗的指标。
bool mse(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
         double *out, std::string *err);

// 10·log10(255² / MSE)。内部就是 mse()，不重复实现一遍公式。
bool psnr(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
          double *out, std::string *err);

// SSIM + MS-SSIM。
bool ssim(const ImageBuffer &a, const ImageBuffer &b, const Options &options,
          SsimResult *out, std::string *err);

// CPU 参考实现（同一套定义，用于自检与无 GPU 场景）。签名与上面一致。
namespace reference {
bool mse(const ImageBuffer &a, const ImageBuffer &b, double *out,
         std::string *err);
bool ssim(const ImageBuffer &a, const ImageBuffer &b, SsimResult *out,
          std::string *err);
} // namespace reference

} // namespace metrics
} // namespace cuda
} // namespace image_codec
