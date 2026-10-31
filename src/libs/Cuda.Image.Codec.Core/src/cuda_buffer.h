/*
 * File: cuda_buffer.h
 * Brief: Cuda buffer.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_buffer.h - 设备内存 RAII 缓冲。
//
// 只做「设备内存」一件事：不做 host<->device 拷贝策略决策。
// 拷贝策略属于各 codec（同步拷贝 / 流式异步 + 事件）而不是容器——但**流**必须
// 由调用方显式给出，原因见下面 upload/download 的注释。
//
// 显存记账：绑定 MemoryBudget 后，每次真实分配都会先记账、释放时归还，
// 因此「本句柄占了多少、峰值多少」是真实的，不是估算。
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <utility>

#include "cuda_error.h"
#include "cuda_memory.h"

namespace image_codec {
namespace cuda {

// 以字节为单位分配的设备缓冲（元素无关，便于同一缓冲复用作不同类型）。
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr_(other.ptr_), bytes_(other.bytes_), budget_(other.budget_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    other.budget_ = nullptr;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      budget_ = other.budget_;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
      other.budget_ = nullptr;
    }
    return *this;
  }

  // 绑定显存预算（nullptr = 不记账）。必须在任何分配之前调用。
  void setBudget(MemoryBudget *budget) { budget_ = budget; }
  MemoryBudget *budget() const { return budget_; }

  // 确保容量 >= bytes（已足够时不重新分配，避免热路径反复 cudaMalloc）。
  bool ensure(size_t bytes, std::string *err) {
    if (bytes <= bytes_ && ptr_ != nullptr) {
      return true;
    }
    if (bytes == 0) {
      return true;
    }
    if (budget_ != nullptr && !budget_->reserve(bytes, err)) {
      return false;
    }
    void *p = nullptr;
    if (!check(cudaMalloc(&p, bytes), "cudaMalloc", err)) {
      if (budget_ != nullptr) budget_->release(bytes);
      return false;
    }
    reset();
    ptr_ = p;
    bytes_ = bytes;
    return true;
  }

  // 按元素数量 ensure（不会改变已有更大容量）。
  template <typename T> bool ensureItems(size_t count, std::string *err) {
    if (count > (static_cast<size_t>(-1)) / sizeof(T)) {
      if (err != nullptr) *err = "device buffer size overflow";
      return false;
    }
    return ensure(count * sizeof(T), err);
  }

  void reset() {
    if (ptr_ != nullptr) {
      // 析构/重建期间不关心返回值，但要清掉可能残留的 sticky error。
      (void)cudaFree(ptr_);
      if (budget_ != nullptr) budget_->release(bytes_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  void *data() const { return ptr_; }
  size_t bytes() const { return bytes_; }
  bool valid() const { return ptr_ != nullptr; }

  template <typename T> T *as() const { return static_cast<T *>(ptr_); }

  // host -> device。
  //
  // stream 必须与「随后消费这份数据的 kernel」所在流一致。默认值 nullptr 是
  // legacy default stream；它和 cudaStreamNonBlocking 创建的流之间**没有**
  // 隐式顺序（只有 blocking stream 才与 legacy default stream 隐式同步）。
  // 曾经因此出过真实缺陷：PNG 自适应滤波在非阻塞流上启动 kernel，而上传走的是
  // legacy default stream，kernel 于是读到还没落地的数据，同一张图每次编码的
  // 结果都不一样（随机垃圾行 -> zlib 输出在 2 KB 与 100 KB 之间跳变）。
  // 这里统一用 cudaMemcpyAsync + 同流同步：语义仍是「返回时数据已就位」，
  // 但对任何流都成立。
  bool upload(const void *src, size_t bytes, std::string *err,
              cudaStream_t stream = nullptr) {
    if (bytes == 0) {
      return true;
    }
    if (!ensure(bytes, err)) {
      return false;
    }
    if (!check(
            cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D", err)) {
      return false;
    }
    return check(cudaStreamSynchronize(stream), "cudaStreamSynchronize", err);
  }

  // device -> host（返回时数据已在 dst）。
  bool download(void *dst, size_t bytes, std::string *err,
                cudaStream_t stream = nullptr) const {
    if (bytes == 0) {
      return true;
    }
    if (ptr_ == nullptr) {
      if (err != nullptr) *err = "download: device buffer not allocated";
      return false;
    }
    if (!check(
            cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync D2H", err)) {
      return false;
    }
    return check(cudaStreamSynchronize(stream), "cudaStreamSynchronize", err);
  }

private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
  MemoryBudget *budget_ = nullptr;
};

} // namespace cuda
} // namespace image_codec
