/*
 * File: cuda_memory.cpp
 * Brief: Cuda memory.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_memory.cpp - 显存查询与预算记账实现。
#include "cuda_memory.h"

#include <cuda_runtime_api.h>

#include "cuda_error.h"

namespace image_codec {
namespace cuda {

bool queryDeviceMemory(int device_index, DeviceMemoryInfo *out,
                       std::string *err) {
  if (out == nullptr) {
    if (err != nullptr) *err = "queryDeviceMemory: null output";
    return false;
  }
  int previous = 0;
  const bool switched = device_index >= 0;
  if (switched) {
    if (!check(cudaGetDevice(&previous), "cudaGetDevice", err)) {
      return false;
    }
    if (!check(cudaSetDevice(device_index), "cudaSetDevice", err)) {
      return false;
    }
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  const cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (switched) {
    (void)cudaSetDevice(previous);
  }
  if (!check(rc, "cudaMemGetInfo", err)) {
    return false;
  }
  out->total_bytes = total_bytes;
  out->free_bytes = free_bytes;
  out->used_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
  return true;
}

void MemoryBudget::setLimit(size_t limit_bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  limit_ = limit_bytes;
}

size_t MemoryBudget::limit() const {
  std::lock_guard<std::mutex> lock(mu_);
  return limit_;
}

void MemoryBudget::bindDevice(int device_index) {
  std::lock_guard<std::mutex> lock(mu_);
  device_index_ = device_index;
}

size_t MemoryBudget::used() const {
  std::lock_guard<std::mutex> lock(mu_);
  return used_;
}

size_t MemoryBudget::peak() const {
  std::lock_guard<std::mutex> lock(mu_);
  return peak_;
}

bool MemoryBudget::reserve(size_t bytes, std::string *err) {
  std::lock_guard<std::mutex> lock(mu_);
  if (bytes == 0) {
    return true;
  }
  const size_t available =
      limit_ == 0 ? 0 : (used_ >= limit_ ? 0 : limit_ - used_);
  if (limit_ != 0 && bytes > available) {
    exhausted_ = true;
    if (err != nullptr) {
      *err = "显存预算不足: 需要 " + std::to_string(bytes) + " 字节, 上限 " +
             std::to_string(limit_) + " 字节, 已用 " + std::to_string(used_) +
             " 字节";
    }
    return false;
  }
  // 大块分配先问驱动：其它进程刚吃掉显存时，最好在这里就失败。
  if (device_index_ >= 0 && bytes >= kProbeThreshold) {
    DeviceMemoryInfo info;
    std::string probe_err;
    if (queryDeviceMemory(device_index_, &info, &probe_err) &&
        bytes > info.free_bytes) {
      exhausted_ = true;
      if (err != nullptr) {
        *err = "设备空闲显存不足: 需要 " + std::to_string(bytes) +
               " 字节, 空闲 " + std::to_string(info.free_bytes) + " 字节";
      }
      return false;
    }
  }
  used_ += bytes;
  if (used_ > peak_) peak_ = used_;
  return true;
}

bool MemoryBudget::exhausted() const {
  std::lock_guard<std::mutex> lock(mu_);
  return exhausted_;
}

bool MemoryBudget::takeExhausted() {
  std::lock_guard<std::mutex> lock(mu_);
  const bool value = exhausted_;
  exhausted_ = false;
  return value;
}

void MemoryBudget::release(size_t bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  used_ = bytes > used_ ? 0 : used_ - bytes;
}

} // namespace cuda
} // namespace image_codec
