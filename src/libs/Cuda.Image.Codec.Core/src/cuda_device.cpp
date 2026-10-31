/*
 * File: cuda_device.cpp
 * Brief: Cuda device.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_device.cpp - 设备枚举与句柄实现。
#include "cuda_device.h"

#include "cuda_error.h"

namespace image_codec {
namespace cuda {

std::vector<DeviceInfo> enumerateDevices() {
  std::vector<DeviceInfo> out;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
    (void)cudaGetLastError(); // 清掉 "no CUDA-capable device" 的 sticky 状态
    return out;
  }
  int driver = 0;
  int runtime = 0;
  (void)cudaDriverGetVersion(&driver);
  (void)cudaRuntimeGetVersion(&runtime);
  char buf[128];
  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
      (void)cudaGetLastError();
      continue;
    }
    DeviceInfo info;
    info.index = i;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.total_memory = static_cast<size_t>(prop.totalGlobalMem);
    info.unified_addressing = prop.unifiedAddressing != 0;
    std::snprintf(buf, sizeof(buf), "%d.%d", driver / 1000,
                  (driver % 1000) / 10);
    info.driver_version = buf;
    std::snprintf(buf, sizeof(buf), "%d.%d", runtime / 1000,
                  (runtime % 1000) / 10);
    info.runtime_version = buf;
    out.push_back(std::move(info));
  }
  return out;
}

bool Device::init(int index, std::string *err) {
  release();
  const auto all = enumerateDevices();
  if (all.empty()) {
    if (err != nullptr) *err = "no CUDA device available";
    return false;
  }
  if (index < 0 || static_cast<size_t>(index) >= all.size()) {
    if (err != nullptr) *err = "CUDA device index out of range";
    return false;
  }
  if (!check(cudaSetDevice(index), "cudaSetDevice", err)) {
    return false;
  }
  // 非阻塞 stream：本库在关键点显式 synchronize，避免与调用方默认 stream 互锁。
  if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags", err)) {
    return false;
  }
  info_ = all[static_cast<size_t>(index)];
  initialized_ = true;
  return true;
}

void Device::release() {
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  initialized_ = false;
}

bool Device::synchronize(std::string *err) const {
  return check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", err);
}

} // namespace cuda
} // namespace image_codec
