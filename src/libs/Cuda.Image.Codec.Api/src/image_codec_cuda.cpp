// image_codec_cuda.cpp - 对外 C API 实现（TIFF / PNG / JPEG 统一入口）。
//
// 本层只做三件事：参数校验、格式分派、C 结构体 <-> ImageBuffer 转换。
// 具体实现全部在各自模块里，避免这里长成第二个 codec。
#include "ImageCodecCUDA/image_codec_cuda.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "cuda_device.h"
#include "cuda_memory.h"
#include "image_buffer.h"
#include "jpeg_codec.h"
#include "metrics.h"
#include "png_codec.h"
#include "tiff_codec.h"

using image_codec::cuda::ImageBuffer;

struct image_codec_cuda_codec {
  int device_index = 0;
  bool jpeg_available = false;
  // 本句柄的显存预算：encode/decode 的所有设备分配都从这里记账。
  image_codec::cuda::MemoryBudget budget;
  std::string device_name;
  int compute_major = 0;
  int compute_minor = 0;
  size_t total_memory = 0;
  std::string note;
};

namespace {

// 可移植的 strdup（Windows 没有 POSIX strdup，且这里必须用 malloc 系列，
// 才能与 image_codec_cuda_device_list_free 的 free 配对）。
char *dupString(const char *s) {
  const size_t n = std::strlen(s);
  auto *out = static_cast<char *>(std::malloc(n + 1));
  if (out == nullptr) return nullptr;
  std::memcpy(out, s, n + 1);
  return out;
}

// ImageBuffer -> C 结构（malloc 一份，调用方负责释放）。
image_codec_cuda_status exportImage(const ImageBuffer &src,
                                    image_codec_cuda_image *out) {
  out->width = static_cast<uint32_t>(src.width);
  out->height = static_cast<uint32_t>(src.height);
  out->channels = static_cast<uint32_t>(src.channels);
  out->stride = src.stride();
  out->data_size = src.data.size();
  out->data = static_cast<uint8_t *>(std::malloc(src.data.size()));
  if (out->data == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
  }
  std::memcpy(out->data, src.data.data(), src.data.size());
  return IMAGE_CODEC_CUDA_OK;
}

// 编码参数默认值：JPEG q90、TIFF LZW、PNG zlib 6 自适应滤波。
void applyDefaults(image_codec_cuda_encode_params *p) {
  if (p->jpeg_quality == 0) p->jpeg_quality = 90;
  if (p->tiff_compression == 0) p->tiff_compression = 5;
  if (p->png_compression == 0) p->png_compression = 6;
  if (p->png_filter == 0) p->png_filter = -1;
}

} // namespace

extern "C" {

const char *image_codec_cuda_status_string(image_codec_cuda_status status) {
  switch (status) {
  case IMAGE_CODEC_CUDA_OK:
    return "success";
  case IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT:
    return "invalid argument";
  case IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY:
    return "out of memory";
  case IMAGE_CODEC_CUDA_ERR_BACKEND_UNAVAILABLE:
    return "backend unavailable";
  case IMAGE_CODEC_CUDA_ERR_UNSUPPORTED_FORMAT:
    return "unsupported format";
  case IMAGE_CODEC_CUDA_ERR_DECODE_FAILED:
    return "decode failed";
  case IMAGE_CODEC_CUDA_ERR_ENCODE_FAILED:
    return "encode failed";
  case IMAGE_CODEC_CUDA_ERR_IO:
    return "I/O error";
  case IMAGE_CODEC_CUDA_ERR_INTERNAL:
    return "internal error";
  default:
    return "unknown status";
  }
}

image_codec_cuda_status
image_codec_cuda_codec_create(image_codec_cuda_codec **out_codec,
                              int device_index) {
  if (out_codec == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  *out_codec = nullptr;
  const auto devices = image_codec::cuda::enumerateDevices();
  if (devices.empty()) {
    return IMAGE_CODEC_CUDA_ERR_BACKEND_UNAVAILABLE;
  }
  if (device_index < 0 || static_cast<size_t>(device_index) >= devices.size()) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  auto *codec = new (std::nothrow) image_codec_cuda_codec();
  if (codec == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
  }
  const auto &info = devices[static_cast<size_t>(device_index)];
  codec->device_index = device_index;
  codec->budget.bindDevice(device_index);
  codec->device_name = info.name;
  codec->compute_major = info.major;
  codec->compute_minor = info.minor;
  codec->total_memory = info.total_memory;
  codec->jpeg_available = image_codec::cuda::jpeg::available();
  if (!codec->jpeg_available) {
    codec->note = "JPEG unavailable: nvJPEG not linked in this build "
                  "(ICUDA_ENABLE_NVJPEG)";
  }
  *out_codec = codec;
  return IMAGE_CODEC_CUDA_OK;
}

void image_codec_cuda_codec_destroy(image_codec_cuda_codec *codec) {
  delete codec;
}

image_codec_cuda_status
image_codec_cuda_get_capabilities(image_codec_cuda_codec *codec,
                                  image_codec_cuda_capabilities *out) {
  if (codec == nullptr || out == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  std::snprintf(out->device_name, sizeof(out->device_name), "%s",
                codec->device_name.c_str());
  out->compute_major = codec->compute_major;
  out->compute_minor = codec->compute_minor;
  out->total_memory = codec->total_memory;
  out->jpeg_decode = codec->jpeg_available ? 1 : 0;
  out->jpeg_encode = codec->jpeg_available ? 1 : 0;
  out->tiff_decode = 1;
  out->tiff_encode = 1;
  out->png_decode = 1;
  out->png_encode = 1;
  out->tiff_lzw_gpu = 1;
  out->png_filter_gpu = 1;
  std::snprintf(out->note, sizeof(out->note), "%s", codec->note.c_str());
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status
image_codec_cuda_device_memory(int device_index,
                               image_codec_cuda_memory_info *out) {
  if (out == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  image_codec::cuda::DeviceMemoryInfo info;
  std::string err;
  if (!image_codec::cuda::queryDeviceMemory(device_index, &info, &err)) {
    return IMAGE_CODEC_CUDA_ERR_BACKEND_UNAVAILABLE;
  }
  out->total_bytes = info.total_bytes;
  out->free_bytes = info.free_bytes;
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status
image_codec_cuda_get_memory_info(image_codec_cuda_codec *codec,
                                 image_codec_cuda_memory_info *out) {
  if (codec == nullptr || out == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  image_codec::cuda::DeviceMemoryInfo info;
  std::string err;
  if (image_codec::cuda::queryDeviceMemory(codec->device_index, &info, &err)) {
    out->total_bytes = info.total_bytes;
    out->free_bytes = info.free_bytes;
  }
  out->budget_bytes = codec->budget.limit();
  out->used_bytes = codec->budget.used();
  out->peak_bytes = codec->budget.peak();
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status
image_codec_cuda_set_memory_limit(image_codec_cuda_codec *codec,
                                  size_t limit_bytes) {
  if (codec == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  codec->budget.setLimit(limit_bytes);
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status image_codec_cuda_probe(const void *data, size_t size,
                                               int *out_format) {
  if (out_format == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  *out_format = IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
  if (data == nullptr || size < 4) {
    return IMAGE_CODEC_CUDA_OK;
  }
  const auto *p = static_cast<const uint8_t *>(data);
  if (image_codec::cuda::png::hasPngSignature(p, size)) {
    *out_format = IMAGE_CODEC_CUDA_FORMAT_PNG;
  } else if (image_codec::cuda::tiff::hasTiffSignature(p, size)) {
    *out_format = IMAGE_CODEC_CUDA_FORMAT_TIFF;
  } else if (image_codec::cuda::jpeg::hasJpegSignature(p, size)) {
    *out_format = IMAGE_CODEC_CUDA_FORMAT_JPEG;
  }
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status image_codec_cuda_decode(image_codec_cuda_codec *codec,
                                                int format, const void *in,
                                                size_t in_len,
                                                image_codec_cuda_image *out) {
  if (codec == nullptr || out == nullptr || in == nullptr || in_len == 0) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  int fmt = format;
  if (fmt == IMAGE_CODEC_CUDA_FORMAT_UNKNOWN) {
    image_codec_cuda_probe(in, in_len, &fmt);
  }
  const auto *bytes = static_cast<const uint8_t *>(in);
  ImageBuffer img;
  std::string err;
  bool ok = false;
  switch (fmt) {
  case IMAGE_CODEC_CUDA_FORMAT_TIFF:
    ok = image_codec::cuda::tiff::decodeTiff(bytes, in_len, &img, &err);
    break;
  case IMAGE_CODEC_CUDA_FORMAT_PNG:
    ok = image_codec::cuda::png::decodePng(bytes, in_len, &img, &err);
    break;
  case IMAGE_CODEC_CUDA_FORMAT_JPEG:
    ok = image_codec::cuda::jpeg::decode(codec->device_index, bytes, in_len,
                                         &img, &err, &codec->budget);
    break;
  default:
    return IMAGE_CODEC_CUDA_ERR_UNSUPPORTED_FORMAT;
  }
  if (!ok) {
    return err.find("nvJPEG") != std::string::npos
               ? IMAGE_CODEC_CUDA_ERR_UNSUPPORTED_FORMAT
               : IMAGE_CODEC_CUDA_ERR_DECODE_FAILED;
  }
  return exportImage(img, out);
}

image_codec_cuda_status
image_codec_cuda_encode(image_codec_cuda_codec *codec, int format,
                        const image_codec_cuda_image *img,
                        const image_codec_cuda_encode_params *params, void *out,
                        size_t *out_len, size_t out_capacity) {
  if (codec == nullptr || img == nullptr || out_len == nullptr ||
      img->data == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (img->width == 0 || img->height == 0 || img->channels == 0 ||
      img->channels > 4) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  ImageBuffer src;
  src.width = static_cast<int>(img->width);
  src.height = static_cast<int>(img->height);
  src.channels = static_cast<int>(img->channels);
  src.data.assign(img->data, img->data + img->data_size);

  image_codec_cuda_encode_params p{};
  if (params != nullptr) {
    p = *params;
  }
  applyDefaults(&p);

  std::vector<uint8_t> encoded;
  std::string err;
  bool ok = false;
  switch (format) {
  case IMAGE_CODEC_CUDA_FORMAT_TIFF: {
    image_codec::cuda::tiff::TiffEncodeOptions opt;
    opt.compression = p.tiff_compression;
    opt.rows_per_strip = p.tiff_rows_per_strip;
    opt.tile_size = p.tiff_tile_size;
    opt.budget = &codec->budget;
    opt.lzw_block_threads = p.tiff_lzw_block_threads;
    opt.lzw_unit_batch = p.tiff_lzw_unit_batch;
    bool gpu_used = false;
    ok = image_codec::cuda::tiff::encodeTiff(src, opt, &encoded, &gpu_used,
                                             &err);
    break;
  }
  case IMAGE_CODEC_CUDA_FORMAT_PNG: {
    image_codec::cuda::png::PngEncodeOptions opt;
    opt.compression_level = p.png_compression;
    opt.filter = p.png_filter;
    opt.block_threads = p.png_block_threads;
    opt.budget = &codec->budget;
    bool gpu_used = false;
    ok = image_codec::cuda::png::encodePng(src, opt, &encoded, &gpu_used, &err);
    break;
  }
  case IMAGE_CODEC_CUDA_FORMAT_JPEG:
    ok = image_codec::cuda::jpeg::encode(codec->device_index, src,
                                         p.jpeg_quality, p.jpeg_subsampling,
                                         &encoded, &err, &codec->budget);
    break;
  default:
    return IMAGE_CODEC_CUDA_ERR_UNSUPPORTED_FORMAT;
  }
  if (!ok) {
    return err.find("nvJPEG") != std::string::npos
               ? IMAGE_CODEC_CUDA_ERR_UNSUPPORTED_FORMAT
               : IMAGE_CODEC_CUDA_ERR_ENCODE_FAILED;
  }

  *out_len = encoded.size();
  if (out == nullptr) {
    // 先查后写：返回所需大小，调用方按此重新调用。
    return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
  }
  if (out_capacity < encoded.size()) {
    return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
  }
  std::memcpy(out, encoded.data(), encoded.size());
  return IMAGE_CODEC_CUDA_OK;
}

namespace {

// C 图像 -> 内部 ImageBuffer。**这一步必然复制整幅图**：C ABI 的 image 由调用方
// malloc、没有 stride 概念，而内部需要 vector 所有权。大图上这是 1.9 GB 级别的
// 拷贝，所以调参（Tune）走内部 C++ 入口而不是 C API——那里的计时不该被它污染。
bool toBufferPair(const image_codec_cuda_image *a,
                  const image_codec_cuda_image *b, ImageBuffer *buf_a,
                  ImageBuffer *buf_b) {
  const auto copy = [](const image_codec_cuda_image *img, ImageBuffer *out) {
    out->width = static_cast<int>(img->width);
    out->height = static_cast<int>(img->height);
    out->channels = static_cast<int>(img->channels);
    out->data.assign(img->data, img->data + img->data_size);
    return out->valid();
  };
  return copy(a, buf_a) && copy(b, buf_b);
}

bool metricArgsValid(const image_codec_cuda_codec *codec,
                     const image_codec_cuda_image *a,
                     const image_codec_cuda_image *b) {
  if (codec == nullptr || a == nullptr || b == nullptr) return false;
  if (a->data == nullptr || b->data == nullptr) return false;
  if (a->width == 0 || a->height == 0 || a->channels == 0) return false;
  return a->width == b->width && a->height == b->height &&
         a->channels == b->channels;
}

} // namespace

// 注意：下面三个入口都会把 C 结构体复制成 ImageBuffer（见 toBufferPair 的注释），
// 失败时也只返回状态码——C API 与 ImageCodecOpenCL 一样不带错误字符串。需要
// 具体原因（例如「显存上限太低」）时走内部 C++ 入口，它带 std::string* err。

image_codec_cuda_status image_codec_cuda_metric_mse(
    image_codec_cuda_codec *codec, const image_codec_cuda_image *a,
    const image_codec_cuda_image *b, double *out_value) {
  if (out_value == nullptr || !metricArgsValid(codec, a, b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  ImageBuffer buf_a;
  ImageBuffer buf_b;
  if (!toBufferPair(a, b, &buf_a, &buf_b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  image_codec::cuda::metrics::Options opt;
  opt.device_index = codec->device_index;
  opt.memory_limit = codec->budget.limit();
  std::string err;
  if (!image_codec::cuda::metrics::mse(buf_a, buf_b, opt, out_value, &err)) {
    return IMAGE_CODEC_CUDA_ERR_INTERNAL;
  }
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status image_codec_cuda_metric_psnr(
    image_codec_cuda_codec *codec, const image_codec_cuda_image *a,
    const image_codec_cuda_image *b, double *out_value) {
  if (out_value == nullptr || !metricArgsValid(codec, a, b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  ImageBuffer buf_a;
  ImageBuffer buf_b;
  if (!toBufferPair(a, b, &buf_a, &buf_b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  image_codec::cuda::metrics::Options opt;
  opt.device_index = codec->device_index;
  opt.memory_limit = codec->budget.limit();
  std::string err;
  if (!image_codec::cuda::metrics::psnr(buf_a, buf_b, opt, out_value, &err)) {
    return IMAGE_CODEC_CUDA_ERR_INTERNAL;
  }
  return IMAGE_CODEC_CUDA_OK;
}

image_codec_cuda_status image_codec_cuda_metric_ssim(
    image_codec_cuda_codec *codec, const image_codec_cuda_image *a,
    const image_codec_cuda_image *b, image_codec_cuda_ssim *out) {
  if (out == nullptr || !metricArgsValid(codec, a, b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  ImageBuffer buf_a;
  ImageBuffer buf_b;
  if (!toBufferPair(a, b, &buf_a, &buf_b)) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  image_codec::cuda::metrics::Options opt;
  opt.device_index = codec->device_index;
  opt.memory_limit = codec->budget.limit();
  image_codec::cuda::metrics::SsimResult r;
  std::string err;
  if (!image_codec::cuda::metrics::ssim(buf_a, buf_b, opt, &r, &err)) {
    return IMAGE_CODEC_CUDA_ERR_INTERNAL;
  }
  out->ssim = r.ssim;
  out->ms_ssim = r.ms_ssim;
  out->levels = r.levels;
  return IMAGE_CODEC_CUDA_OK;
}

void image_codec_cuda_image_free(image_codec_cuda_image *img) {
  if (img == nullptr) return;
  std::free(img->data);
  std::memset(img, 0, sizeof(*img));
}

image_codec_cuda_status image_codec_cuda_device_list(char ***names,
                                                     int *count) {
  if (names == nullptr || count == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_INVALID_ARGUMENT;
  }
  *names = nullptr;
  *count = 0;
  const auto devices = image_codec::cuda::enumerateDevices();
  if (devices.empty()) {
    return IMAGE_CODEC_CUDA_OK;
  }
  auto **list =
      static_cast<char **>(std::calloc(devices.size(), sizeof(char *)));
  if (list == nullptr) {
    return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
  }
  size_t written = 0;
  for (const auto &d : devices) {
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%d: %s (sm_%d%d, %zu MiB)",
                  static_cast<int>(written), d.name.c_str(), d.major, d.minor,
                  d.total_memory >> 20);
    list[written] = dupString(buf);
    if (list[written] == nullptr) {
      for (size_t i = 0; i < written; ++i) std::free(list[i]);
      std::free(list);
      return IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY;
    }
    ++written;
  }
  *names = list;
  *count = static_cast<int>(written);
  return IMAGE_CODEC_CUDA_OK;
}

void image_codec_cuda_device_list_free(char **names, int count) {
  if (names == nullptr) return;
  for (int i = 0; i < count; ++i) std::free(names[i]);
  std::free(names);
}

} // extern "C"
