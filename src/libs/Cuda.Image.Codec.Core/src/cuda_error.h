/*
 * File: cuda_error.h
 * Brief: Cuda error.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// cuda_error.h - CUDA 错误到可读文本的统一转换（带调用点上下文）。
#pragma once

#include <cuda_runtime_api.h>

#include <string>

namespace image_codec {
namespace cuda {

// cudaError_t -> "cudaErrorXxx: 描述"。
std::string errorText(cudaError_t e);

// 失败时写 err = "<what>: <errorText>"，返回 false；成功返回 true。
inline bool check(cudaError_t e, const char *what, std::string *err) {
  if (e == cudaSuccess) {
    return true;
  }
  if (err != nullptr) {
    *err =
        std::string(what != nullptr ? what : "cuda call") + ": " + errorText(e);
  }
  return false;
}

// 记录错误并清空 CUDA sticky 错误（诊断用，不吞掉最后一次失败信息）。
void noteError(const char *what, cudaError_t e);

} // namespace cuda
} // namespace image_codec
