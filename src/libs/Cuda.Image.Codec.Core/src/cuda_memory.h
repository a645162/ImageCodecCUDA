/*
 * File: cuda_memory.h
 * Brief: Cuda memory.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_memory.h - 显存查询与「显存上限」记账。
//
// 为什么需要单独的显存记账而不是直接看 cudaMemGetInfo：
//   * cudaMemGetInfo 的 free 是「驱动视角」，其它进程（浏览器、训练进程、
//     ImageViewer 自己的 CUDA-GL 互操作）随时会把它吃掉，编码器按它分配
//     会在中途 OOM；
//   * 大图（本机 Merge.bmp 就有 1.89 GB）在显存有限的机器上必须能被
//     「限流」而不是把整机拖死，因此本库允许调用方给一个硬上限；
//   * 上层 router 需要知道「本句柄到底占了多少、峰值多少」，才能决定是否
//     继续把下一张图交给 CUDA 后端。
#pragma once

#include <cstddef>
#include <mutex>
#include <string>

namespace image_codec {
namespace cuda {

// 设备显存快照（字节）。
struct DeviceMemoryInfo {
  size_t total_bytes = 0;
  size_t free_bytes = 0;
  size_t used_bytes = 0; // total - free
};

// 查询指定设备的显存。device_index < 0 时查当前设备。
// 无设备/驱动不匹配时返回 false 并写 err（不会打印）。
bool queryDeviceMemory(int device_index, DeviceMemoryInfo *out,
                       std::string *err);

// 本句柄的显存预算。
//
// 语义：
//   * limit == 0 表示「不设上限」，此时只受设备真实空闲显存约束；
//   * reserve() 在两种情况下失败：超过 limit，或在 device 已绑定时
//     「本次分配 > 当前空闲显存」（提前失败，给出可读原因，
//     而不是让 cudaMalloc 抛一个没有上下文的错误码）；
//   * 所有计数都是字节，线程安全（同一 Codec 句柄可能被多线程复用）。
class MemoryBudget {
public:
  MemoryBudget() = default;
  MemoryBudget(const MemoryBudget &) = delete;
  MemoryBudget &operator=(const MemoryBudget &) = delete;

  // limit_bytes == 0 = 不限制。
  void setLimit(size_t limit_bytes);
  size_t limit() const;

  // 绑定设备后，reserve() 会额外校验设备空闲显存。device_index < 0 解绑。
  void bindDevice(int device_index);

  size_t used() const;
  size_t peak() const;

  // 记账并预占 bytes。失败时 used() 不变。
  bool reserve(size_t bytes, std::string *err);

  // 归还 bytes（必须与成功的 reserve 一一对应）。
  void release(size_t bytes);

  // 上一次 reserve() 是否因「预算/设备空闲显存不足」失败。
  //
  // 存在的意义：调用方（C API）要区分「这台设备的资源不够」与「图真的编坏了」。
  // 前者换个后端还有救，后者换个后端只会把 bug 藏起来。只在 reserve() 里能
  // 得到的知识，就不要让上层去猜错误字符串。
  bool exhausted() const;

  // 读取并清除（调用方在每次操作前先清一次，避免读到上一次的结论）。
  bool takeExhausted();

private:
  // 设备空闲显存的提前校验只对「够大」的分配做：小额分配（几千字节）
  // 每次都问驱动是纯浪费，真正的 OOM 风险都在大块上。
  static constexpr size_t kProbeThreshold = 32u << 20; // 32 MiB

  mutable std::mutex mu_;
  size_t limit_ = 0;
  size_t used_ = 0;
  size_t peak_ = 0;
  int device_index_ = -1;
  bool exhausted_ = false;
};

} // namespace cuda
} // namespace image_codec
