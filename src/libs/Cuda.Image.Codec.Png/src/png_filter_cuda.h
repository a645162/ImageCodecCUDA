/*
 * File: png_filter_cuda.h
 * Brief: Png filter cuda.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// png_filter_cuda.h - PNG 行滤波的 CUDA 实现（自适应选择 + 生成 filtered 流）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cuda_buffer.h"
#include "cuda_device.h"
#include "cuda_memory.h"
#include "image_buffer.h"

namespace image_codec {
namespace cuda {
namespace png {

class PngFilterCuda {
public:
  explicit PngFilterCuda(int device_index = 0);
  ~PngFilterCuda();

  PngFilterCuda(const PngFilterCuda &) = delete;
  PngFilterCuda &operator=(const PngFilterCuda &) = delete;

  // 生成 filtered 流：height 行，每行 (1 + width*channels) 字节。
  // filter_type: -1 = 逐行自适应（最小绝对值和），0..4 = 固定滤波器。
  // filters_out 可选：回报每行实际使用的滤波器。
  bool buildFilteredStream(const ImageBuffer &img, int filter_type,
                           std::vector<uint8_t> *out,
                           std::vector<uint8_t> *filters_out);

  const std::string &lastError() const { return err_; }

  // 调参旋钮：一行一个 block，block 越大单行摊销越少但占用越多。
  // <= 0 用默认 256；内部裁剪到 [32,1024] 且按 32 对齐。
  void setBlockThreads(int block_threads) { block_threads_ = block_threads; }
  int blockThreads() const { return block_threads_; }

  // 绑定显存预算（所有设备缓冲的分配都记账）。传 nullptr 表示只用内部记账。
  // 必须在第一次 buildFilteredStream 之前调用（之后改绑会让已有分配失去归属）。
  void setBudget(MemoryBudget *budget);

private:
  bool initOnce();
  void bindBudget(MemoryBudget *budget);

  int device_index_;
  int block_threads_ = 0;
  std::string err_;
  bool init_done_ = false;
  MemoryBudget own_budget_; // 上层未注入预算时使用
  MemoryBudget *budget_ = &own_budget_;
  Device dev_;
  DeviceBuffer src_;
  DeviceBuffer dst_;
  DeviceBuffer types_;
  DeviceBuffer sums_;
};

} // namespace png
} // namespace cuda
} // namespace image_codec
