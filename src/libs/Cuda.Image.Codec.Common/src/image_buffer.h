/*
 * File: image_buffer.h
 * Brief: Image buffer.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// image_buffer.h - 宿主侧图像缓冲（top-down，1/2/3/4 通道）。
//
// 与主项目 ImageViewer 的约定一致：行序 top-down，第 0 行是图像顶部；
// 通道数为实际样本数：1 = 灰度、2 = 灰度+Alpha、3 = RGB、4 = RGBA。
// 与 PNG 的四种颜色类型一一对应；不做隐式升维，也不丢 Alpha。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace image_codec {
namespace cuda {

struct ImageBuffer {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> data;

  size_t stride() const {
    return static_cast<size_t>(width) * static_cast<size_t>(channels);
  }
  size_t bytes() const { return stride() * static_cast<size_t>(height); }
  bool valid() const {
    return width > 0 && height > 0 &&
           (channels == 1 || channels == 2 || channels == 3 || channels == 4) &&
           data.size() == bytes();
  }

  // 按 (w, h, c) 分配并清零；参数非法返回 false。
  bool allocate(int w, int h, int c) {
    if (w <= 0 || h <= 0 || (c != 1 && c != 2 && c != 3 && c != 4)) {
      return false;
    }
    width = w;
    height = h;
    channels = c;
    data.assign(static_cast<size_t>(w) * static_cast<size_t>(h) *
                    static_cast<size_t>(c),
                0);
    return true;
  }
};

} // namespace cuda
} // namespace image_codec
