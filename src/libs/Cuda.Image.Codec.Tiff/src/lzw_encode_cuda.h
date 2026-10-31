/*
 * File: lzw_encode_cuda.h
 * Brief: Lzw encode cuda.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// lzw_encode_cuda.h - TIFF LZW 的 CUDA 编码器（单位并行 kernel）。
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
namespace tiff {

// 把连续的等长单元（strip 或已 padding 的 tile）压成独立的 LZW 流。
// 输出按单元顺序拼接，out_sizes[i] 为第 i 个单元的压缩字节数。
class LzwEncoderCuda {
public:
  explicit LzwEncoderCuda(int device_index = 0);
  ~LzwEncoderCuda();

  LzwEncoderCuda(const LzwEncoderCuda &) = delete;
  LzwEncoderCuda &operator=(const LzwEncoderCuda &) = delete;

  // 按 strip 编码。rows_per_strip 必须整除 height（调用方保证）。
  int encodeStrips(const ImageBuffer &img, int rows_per_strip,
                   std::vector<uint32_t> &out_sizes,
                   std::vector<uint8_t> &out_data);

  // 按 tile 编码。tile_size 建议 256/512；边缘 tile 会按规格 padding 到整块。
  int encodeTiles(const ImageBuffer &img, int tile_size,
                  std::vector<uint32_t> &out_sizes,
                  std::vector<uint8_t> &out_data);

  // 底层：input 是 n_units * unit_bytes 的连续宿主缓冲（单元已在宿主侧拼好，
  // 例如已 padding 的 tile 平面），按批上传后逐单元并行编码。
  int encodePaddedUnits(const uint8_t *input, size_t unit_bytes, size_t n_units,
                        std::vector<uint32_t> &out_sizes,
                        std::vector<uint8_t> &out_data);

  const std::string &lastError() const { return err_; }

  // ---- 调参旋钮（不同 GPU 最优值不同，交给 Tune 库扫描）----
  // 每个 block 的线程数：一个线程处理一个 unit，0 = 默认 128，裁剪到 [32,512]。
  void setBlockThreads(int block_threads) { block_threads_ = block_threads; }
  int blockThreads() const { return block_threads_; }
  // 每批上传多少个 unit（0 = 按显存预算自动）。过大吃显存，过小压不满 GPU。
  void setUnitBatch(int units) { unit_batch_ = units; }
  int unitBatch() const { return unit_batch_; }
  // 绑定显存预算（所有设备缓冲的分配都记账）。须在首次编码前调用。
  void setBudget(MemoryBudget *budget);

  // 上一次编码实际使用的批大小（诊断/调参用）。
  size_t lastBatchUnits() const { return last_batch_units_; }

private:
  bool initOnce();
  void bindBudget(MemoryBudget *budget);
  // 本批允许使用的字节数（预算 / 设备空闲 / 兜底默认值）。
  size_t batchBytesBudget() const;

  int device_index_;
  int block_threads_ = 0;
  int unit_batch_ = 0;
  size_t last_batch_units_ = 0;
  MemoryBudget own_budget_;
  MemoryBudget *budget_ = &own_budget_;
  std::string err_;
  bool init_done_ = false;
  Device dev_;
  DeviceBuffer in_;
  DeviceBuffer dict_;
  DeviceBuffer out_;
  DeviceBuffer sizes_;
};

} // namespace tiff
} // namespace cuda
} // namespace image_codec
