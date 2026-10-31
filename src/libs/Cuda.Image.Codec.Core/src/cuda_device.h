/*
 * File: cuda_device.h
 * Brief: Cuda device.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_device.h - CUDA 设备枚举与 context/stream RAII 句柄。
//
// 本库刻意不使用 CUDA Runtime 的「隐式默认 context 懒初始化」：每个 Codec 唯一持有
// 一个 Device，显式 cudaSetDevice + 显式 stream，避免多设备场景下把 kernel 发到
// 错误设备（ImageViewer 会同时枚举 4070 与核显）。
#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace image_codec {
namespace cuda {

struct DeviceInfo {
  int index = 0;
  std::string name;
  int major = 0; // compute capability major
  int minor = 0;
  size_t total_memory = 0;
  bool unified_addressing = false;
  std::string driver_version;
  std::string runtime_version;
};

// 枚举全部 CUDA 设备。没有任何设备时返回空表（不是错误）。
std::vector<DeviceInfo> enumerateDevices();

// 带显式 stream 的设备句柄。
class Device {
public:
  Device() = default;
  ~Device() { release(); }

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  // 初始化（同进程内可重复调用：同一 index 会复用底层 context）。
  // 失败返回 false 并写 err（不会打印）。
  bool init(int index, std::string *err);

  void release();

  bool valid() const { return initialized_; }
  int index() const { return info_.index; }
  const DeviceInfo &info() const { return info_; }
  cudaStream_t stream() const { return stream_; }

  // 等待本设备 stream 上的全部工作完成。
  bool synchronize(std::string *err) const;

  // 读取实际运行时可用的驱动/运行时版本（init 后填充）。
  const std::string &driverVersion() const { return info_.driver_version; }
  const std::string &runtimeVersion() const { return info_.runtime_version; }

private:
  DeviceInfo info_;
  cudaStream_t stream_ = nullptr;
  bool initialized_ = false;
};

} // namespace cuda
} // namespace image_codec
