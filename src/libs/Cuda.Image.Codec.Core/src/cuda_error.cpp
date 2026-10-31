// cuda_error.cpp - CUDA 错误文本实现。
#include "cuda_error.h"

#include <cstdio>

namespace image_codec {
namespace cuda {

std::string errorText(cudaError_t e) {
  const char *name = cudaGetErrorName(e);
  const char *desc = cudaGetErrorString(e);
  std::string s;
  if (name != nullptr) {
    s += name;
    s += ": ";
  }
  if (desc != nullptr) {
    s += desc;
  }
  if (s.empty()) {
    s = "unknown CUDA error";
  }
  return s;
}

void noteError(const char *what, cudaError_t e) {
  std::fprintf(stderr, "[ImageCodecCUDA] %s: %s\n",
               what != nullptr ? what : "-", errorText(e).c_str());
}

} // namespace cuda
} // namespace image_codec
