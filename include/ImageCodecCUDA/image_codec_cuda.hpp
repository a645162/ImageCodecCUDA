/*
 * File: image_codec_cuda.hpp
 * Brief: Image codec cuda.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// image_codec_cuda.hpp - ImageCodecCUDA 的 C++ RAII 包装。
//
// 只做「资源管理 + 错误转异常」，不引入第二套语义：
//   * 所有行为与 image_codec_cuda.h 完全一致；
//   * 失败抛 image_codec_cuda::Error（携带 status + 上下文文本）。
//
// 不想用异常的程序直接使用 C API；上层 router 使用 C API，因此两者不会分叉。
#pragma once

#include <ImageCodecCUDA/image_codec_cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace image_codec_cuda {

class Error : public std::runtime_error {
public:
  Error(image_codec_cuda_status status, std::string what)
      : std::runtime_error(std::move(what)), status_(status) {}
  image_codec_cuda_status status() const noexcept { return status_; }

private:
  image_codec_cuda_status status_;
};

// 24/8/32 位通用图像缓冲（top-down）。
struct Image {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<uint8_t> pixels; // size == stride * height

  size_t stride() const { return static_cast<size_t>(width) * channels; }
  bool empty() const { return pixels.empty(); }
};

// 编码参数（默认值与 C 结构体一致：JPEG q90 / TIFF LZW / PNG 自适应滤波）。
struct EncodeParams {
  int jpeg_quality = 90;
  int jpeg_subsampling = 0;
  int tiff_compression = 5;
  int tiff_rows_per_strip = 0;
  int tiff_tile_size = 0;
  int png_compression = 6;
  int png_filter = -1;
};

class Codec {
public:
  explicit Codec(int device_index = 0) {
    const auto st = image_codec_cuda_codec_create(&codec_, device_index);
    if (st != IMAGE_CODEC_CUDA_OK) {
      throw Error(st, std::string("create codec: ") +
                          image_codec_cuda_status_string(st));
    }
  }
  ~Codec() { image_codec_cuda_codec_destroy(codec_); }

  Codec(const Codec &) = delete;
  Codec &operator=(const Codec &) = delete;
  Codec(Codec &&other) noexcept : codec_(other.codec_) {
    other.codec_ = nullptr;
  }
  Codec &operator=(Codec &&other) noexcept {
    if (this != &other) {
      image_codec_cuda_codec_destroy(codec_);
      codec_ = other.codec_;
      other.codec_ = nullptr;
    }
    return *this;
  }

  image_codec_cuda_codec *get() const noexcept { return codec_; }

  image_codec_cuda_capabilities capabilities() const {
    image_codec_cuda_capabilities caps{};
    const auto st = image_codec_cuda_get_capabilities(codec_, &caps);
    if (st != IMAGE_CODEC_CUDA_OK) {
      throw Error(st, std::string("capabilities: ") +
                          image_codec_cuda_status_string(st));
    }
    return caps;
  }

  Image decode(int format, const void *data, size_t size) const {
    image_codec_cuda_image out{};
    const auto st = image_codec_cuda_decode(codec_, format, data, size, &out);
    if (st != IMAGE_CODEC_CUDA_OK) {
      throw Error(st,
                  std::string("decode: ") + image_codec_cuda_status_string(st));
    }
    Image img;
    img.width = out.width;
    img.height = out.height;
    img.channels = out.channels;
    img.pixels.assign(out.data, out.data + out.data_size);
    image_codec_cuda_image_free(&out);
    return img;
  }

  std::vector<uint8_t> encode(int format, const Image &img,
                              const EncodeParams &params = {}) const {
    image_codec_cuda_image in{};
    in.width = img.width;
    in.height = img.height;
    in.channels = img.channels;
    in.data = const_cast<uint8_t *>(img.pixels.data());
    in.stride = img.stride();
    in.data_size = img.pixels.size();

    image_codec_cuda_encode_params p{};
    p.jpeg_quality = params.jpeg_quality;
    p.jpeg_subsampling = params.jpeg_subsampling;
    p.tiff_compression = params.tiff_compression;
    p.tiff_rows_per_strip = params.tiff_rows_per_strip;
    p.tiff_tile_size = params.tiff_tile_size;
    p.png_compression = params.png_compression;
    p.png_filter = params.png_filter;

    size_t need = 0;
    auto st =
        image_codec_cuda_encode(codec_, format, &in, &p, nullptr, &need, 0);
    if (st != IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY && st != IMAGE_CODEC_CUDA_OK) {
      throw Error(st, std::string("encode(size): ") +
                          image_codec_cuda_status_string(st));
    }
    std::vector<uint8_t> out(need);
    size_t len = need;
    st = image_codec_cuda_encode(codec_, format, &in, &p, out.data(), &len,
                                 out.size());
    if (st != IMAGE_CODEC_CUDA_OK) {
      throw Error(st,
                  std::string("encode: ") + image_codec_cuda_status_string(st));
    }
    out.resize(len);
    return out;
  }

private:
  image_codec_cuda_codec *codec_ = nullptr;
};

// 设备名列表（RAII）。
inline std::vector<std::string> deviceList() {
  char **names = nullptr;
  int count = 0;
  const auto st = image_codec_cuda_device_list(&names, &count);
  std::vector<std::string> out;
  if (st == IMAGE_CODEC_CUDA_OK) {
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) out.emplace_back(names[i]);
  }
  image_codec_cuda_device_list_free(names, count);
  return out;
}

// 魔数探测。
inline int probeFormat(const void *data, size_t size) {
  int fmt = IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
  image_codec_cuda_probe(data, size, &fmt);
  return fmt;
}

} // namespace image_codec_cuda
