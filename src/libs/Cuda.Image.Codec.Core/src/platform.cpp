/*
 * File: platform.cpp
 * Brief: Platform.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// platform.cpp - 宿主侧小工具实现（C++20 chrono，无平台 ifdef）。
#include "platform.h"

#include <chrono>
#include <thread>

namespace image_codec {
namespace cuda {

long long nowNs() {
  const auto t = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

void sleepMs(unsigned ms) {
  if (ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
}

} // namespace cuda
} // namespace image_codec
