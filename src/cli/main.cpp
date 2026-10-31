// src/cli/main.cpp - ImageCodecCUDA 测试 CLI（list-devices / info / decode / encode / bench）。
//
// 只做文件 I/O 与参数解析：格式探测与编解码全部走 C API，保证 CLI 与库行为一致。
#include <ImageCodecCUDA/image_codec_cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bmp_io.h"

namespace {

const char *kUsage =
    "用法:\n"
    "  icu list-devices\n"
    "  icu info <file>\n"
    "  icu decode <in> <out.bmp> [--device N]\n"
    "  icu encode <in.bmp> <out> [--format jpeg|tiff|png] [--quality N]\n"
    "             [--subsampling 444|420|422] [--tiff-compression 1|5]\n"
    "             [--tile N] [--rows-per-strip N] [--png-filter -1..4]\n"
    "             [--png-compression 0-9] [--device N]\n"
    "  icu mem [--device N] [--limit-mb N]\n"
    "  icu compare <a.bmp> <b.bmp> [--device N]   # MSE / PSNR / SSIM / "
    "MS-SSIM\n"
    "  icu bench <file> [--iterations N] [--device N]\n"
    "\n"
    "通用选项: --device N（默认 0）、--limit-mb N（显存上限，0 = 不限制）\n";

bool readFile(const char *path, std::vector<uint8_t> *out) {
  FILE *f = std::fopen(path, "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len < 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(len));
  const size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  return got == out->size();
}

bool writeFile(const char *path, const std::vector<uint8_t> &data) {
  FILE *f = std::fopen(path, "wb");
  if (f == nullptr) return false;
  const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
  std::fclose(f);
  return ok;
}

const char *formatName(int fmt) {
  switch (fmt) {
  case IMAGE_CODEC_CUDA_FORMAT_JPEG:
    return "jpeg";
  case IMAGE_CODEC_CUDA_FORMAT_TIFF:
    return "tiff";
  case IMAGE_CODEC_CUDA_FORMAT_PNG:
    return "png";
  default:
    return "unknown";
  }
}

int formatFromName(const char *s) {
  if (std::strcmp(s, "jpeg") == 0 || std::strcmp(s, "jpg") == 0) {
    return IMAGE_CODEC_CUDA_FORMAT_JPEG;
  }
  if (std::strcmp(s, "tiff") == 0 || std::strcmp(s, "tif") == 0) {
    return IMAGE_CODEC_CUDA_FORMAT_TIFF;
  }
  if (std::strcmp(s, "png") == 0) {
    return IMAGE_CODEC_CUDA_FORMAT_PNG;
  }
  return IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
}

// --limit-mb：所有命令共用（大图在小显存机器上必须能限流，而不是直接 OOM）。
size_t gLimitBytes = 0;

// 创建 codec 并套用显存上限；失败返回 nullptr。
image_codec_cuda_codec *createCodec(int device) {
  image_codec_cuda_codec *codec = nullptr;
  if (image_codec_cuda_codec_create(&codec, device) != IMAGE_CODEC_CUDA_OK) {
    return nullptr;
  }
  if (gLimitBytes > 0) {
    image_codec_cuda_set_memory_limit(codec, gLimitBytes);
  }
  return codec;
}

// 24-bit BMP -> RGB top-down 的 C 图像（malloc，调用方 free data）。
bool loadBmpAsImage(const char *path, image_codec_cuda_image *out) {
  image_codec::cuda::BmpImage bmp;
  if (image_codec::cuda::bmpLoad(path, bmp) != 0) return false;

  std::memset(out, 0, sizeof(*out));
  out->width = static_cast<uint32_t>(bmp.width);
  out->height = static_cast<uint32_t>(bmp.height);
  out->channels = 3;
  out->stride = static_cast<size_t>(bmp.width) * 3;
  out->data_size = out->stride * static_cast<size_t>(bmp.height);
  out->data = static_cast<uint8_t *>(std::malloc(out->data_size));
  if (out->data == nullptr) return false;

  // BMP 是 BGR bottom-up，bmpLoad 已翻成 top-down BGR，这里只换通道序。
  const size_t pixels =
      static_cast<size_t>(bmp.width) * static_cast<size_t>(bmp.height);
  for (size_t i = 0; i < pixels; ++i) {
    out->data[i * 3 + 0] = bmp.bgr[i * 3 + 2];
    out->data[i * 3 + 1] = bmp.bgr[i * 3 + 1];
    out->data[i * 3 + 2] = bmp.bgr[i * 3 + 0];
  }
  return true;
}

int cmdListDevices() {
  char **names = nullptr;
  int count = 0;
  const auto st = image_codec_cuda_device_list(&names, &count);
  if (st != IMAGE_CODEC_CUDA_OK) {
    std::fprintf(stderr, "list-devices failed: %s\n",
                 image_codec_cuda_status_string(st));
    return 1;
  }
  for (int i = 0; i < count; ++i) {
    std::printf("%s\n", names[i]);
  }
  image_codec_cuda_device_list_free(names, count);
  return 0;
}

int cmdInfo(const char *path, int device) {
  std::vector<uint8_t> bytes;
  if (!readFile(path, &bytes)) {
    std::fprintf(stderr, "无法读取: %s\n", path);
    return 1;
  }
  int fmt = IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
  image_codec_cuda_probe(bytes.data(), bytes.size(), &fmt);
  std::printf("file    : %s\n", path);
  std::printf("size    : %zu bytes\n", bytes.size());
  std::printf("format  : %s\n", formatName(fmt));
  if (fmt == IMAGE_CODEC_CUDA_FORMAT_UNKNOWN) {
    return 2;
  }
  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败（无 CUDA 设备？）\n");
    return 3;
  }
  image_codec_cuda_image img{};
  const auto st =
      image_codec_cuda_decode(codec, fmt, bytes.data(), bytes.size(), &img);
  if (st == IMAGE_CODEC_CUDA_OK) {
    std::printf("decode  : %ux%u x%u\n", img.width, img.height, img.channels);
    image_codec_cuda_image_free(&img);
  } else {
    std::printf("decode  : failed (%s)\n", image_codec_cuda_status_string(st));
  }
  image_codec_cuda_codec_destroy(codec);
  return st == IMAGE_CODEC_CUDA_OK ? 0 : 4;
}

int cmdDecode(const char *in, const char *out, int device) {
  std::vector<uint8_t> bytes;
  if (!readFile(in, &bytes)) {
    std::fprintf(stderr, "无法读取: %s\n", in);
    return 1;
  }
  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败（无 CUDA 设备？）\n");
    return 2;
  }
  image_codec_cuda_image img{};
  const auto st = image_codec_cuda_decode(
      codec, IMAGE_CODEC_CUDA_FORMAT_UNKNOWN, bytes.data(), bytes.size(), &img);
  if (st != IMAGE_CODEC_CUDA_OK) {
    std::fprintf(stderr, "decode 失败: %s\n",
                 image_codec_cuda_status_string(st));
    image_codec_cuda_codec_destroy(codec);
    return 3;
  }
  // 只有 3 通道能写 24-bit BMP（灰度复制成 RGB）。
  image_codec::cuda::BmpImage bmp;
  bmp.width = static_cast<int>(img.width);
  bmp.height = static_cast<int>(img.height);
  bmp.bgr.resize(static_cast<size_t>(img.width) * img.height * 3);
  const size_t pixels = static_cast<size_t>(img.width) * img.height;
  for (size_t i = 0; i < pixels; ++i) {
    if (img.channels == 1) {
      const uint8_t v = img.data[i];
      bmp.bgr[i * 3 + 0] = v;
      bmp.bgr[i * 3 + 1] = v;
      bmp.bgr[i * 3 + 2] = v;
    } else if (img.channels == 4) {
      bmp.bgr[i * 3 + 0] = img.data[i * 4 + 2];
      bmp.bgr[i * 3 + 1] = img.data[i * 4 + 1];
      bmp.bgr[i * 3 + 2] = img.data[i * 4 + 0];
    } else {
      bmp.bgr[i * 3 + 0] = img.data[i * 3 + 2];
      bmp.bgr[i * 3 + 1] = img.data[i * 3 + 1];
      bmp.bgr[i * 3 + 2] = img.data[i * 3 + 0];
    }
  }
  image_codec_cuda_image_free(&img);
  image_codec_cuda_codec_destroy(codec);
  if (image_codec::cuda::bmpSave(out, bmp) != 0) {
    std::fprintf(stderr, "写 BMP 失败: %s\n", out);
    return 4;
  }
  return 0;
}

int cmdEncode(const char *in, const char *out, int fmt, int device,
              const image_codec_cuda_encode_params &params) {
  image_codec_cuda_image img{};
  if (!loadBmpAsImage(in, &img)) {
    std::fprintf(stderr, "读取 BMP 失败（需要 24-bit）: %s\n", in);
    return 1;
  }

  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败（无 CUDA 设备？）\n");
    std::free(img.data);
    return 2;
  }

  size_t need = 0;
  auto st =
      image_codec_cuda_encode(codec, fmt, &img, &params, nullptr, &need, 0);
  if (st != IMAGE_CODEC_CUDA_ERR_OUT_OF_MEMORY && st != IMAGE_CODEC_CUDA_OK) {
    std::fprintf(stderr, "encode 失败: %s\n",
                 image_codec_cuda_status_string(st));
    std::free(img.data);
    image_codec_cuda_codec_destroy(codec);
    return 4;
  }
  std::vector<uint8_t> encoded(need);
  st = image_codec_cuda_encode(codec, fmt, &img, &params, encoded.data(), &need,
                               encoded.size());
  std::free(img.data);
  image_codec_cuda_codec_destroy(codec);
  if (st != IMAGE_CODEC_CUDA_OK) {
    std::fprintf(stderr, "encode 失败: %s\n",
                 image_codec_cuda_status_string(st));
    return 5;
  }
  encoded.resize(need);
  if (!writeFile(out, encoded)) {
    std::fprintf(stderr, "写文件失败: %s\n", out);
    return 6;
  }
  std::printf("wrote %zu bytes -> %s\n", encoded.size(), out);
  return 0;
}

int cmdBench(const char *in, int iterations, int device) {
  std::vector<uint8_t> bytes;
  if (!readFile(in, &bytes)) {
    std::fprintf(stderr, "无法读取: %s\n", in);
    return 1;
  }
  int fmt = IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
  image_codec_cuda_probe(bytes.data(), bytes.size(), &fmt);
  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败（无 CUDA 设备？）\n");
    return 2;
  }
  const auto t0 = std::chrono::steady_clock::now();
  image_codec_cuda_image img{};
  double best_ms = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto s = std::chrono::steady_clock::now();
    image_codec_cuda_image tmp{};
    const auto st =
        image_codec_cuda_decode(codec, fmt, bytes.data(), bytes.size(), &tmp);
    const auto e = std::chrono::steady_clock::now();
    if (st != IMAGE_CODEC_CUDA_OK) {
      std::fprintf(stderr, "decode 失败: %s\n",
                   image_codec_cuda_status_string(st));
      image_codec_cuda_codec_destroy(codec);
      return 3;
    }
    const double ms = std::chrono::duration<double, std::milli>(e - s).count();
    if (i == 0 || ms < best_ms) best_ms = ms;
    if (i == 0) img = tmp;
    else image_codec_cuda_image_free(&tmp);
  }
  const double total_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
  std::printf("decode %s %ux%u: best %.2f ms, avg %.2f ms (%d iters)\n",
              formatName(fmt), img.width, img.height, best_ms,
              total_ms / iterations, iterations);
  image_codec_cuda_image_free(&img);
  image_codec_cuda_codec_destroy(codec);
  return 0;
}

int cmdMem(int device) {
  image_codec_cuda_memory_info mi{};
  const auto st = image_codec_cuda_device_memory(device, &mi);
  if (st != IMAGE_CODEC_CUDA_OK) {
    std::fprintf(stderr, "查询显存失败: %s\n",
                 image_codec_cuda_status_string(st));
    return 2;
  }
  std::printf("device        : %d\n", device);
  std::printf("total         : %zu MiB\n", mi.total_bytes >> 20);
  std::printf("free          : %zu MiB\n", mi.free_bytes >> 20);

  // 句柄侧的记账要创建 codec 才有意义（budget/used/peak 属于句柄而不是设备）。
  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败\n");
    return 3;
  }
  image_codec_cuda_memory_info handle{};
  if (image_codec_cuda_get_memory_info(codec, &handle) == IMAGE_CODEC_CUDA_OK) {
    std::printf("budget        : %zu MiB%s\n", handle.budget_bytes >> 20,
                handle.budget_bytes == 0 ? "（不限制）" : "");
    std::printf("used / peak   : %zu / %zu MiB\n", handle.used_bytes >> 20,
                handle.peak_bytes >> 20);
  }
  image_codec_cuda_codec_destroy(codec);
  return 0;
}

int cmdCompare(const char *path_a, const char *path_b, int device) {
  image_codec_cuda_image a{};
  image_codec_cuda_image b{};
  if (!loadBmpAsImage(path_a, &a) || !loadBmpAsImage(path_b, &b)) {
    std::fprintf(stderr, "读取 BMP 失败（需要 24-bit）\n");
    std::free(a.data);
    std::free(b.data);
    return 1;
  }

  image_codec_cuda_codec *codec = createCodec(device);
  if (codec == nullptr) {
    std::fprintf(stderr, "创建 codec 失败（无 CUDA 设备？）\n");
    std::free(a.data);
    std::free(b.data);
    return 2;
  }

  int rc = 0;
  const auto show = [&](const char *name, image_codec_cuda_status st) {
    if (st != IMAGE_CODEC_CUDA_OK) {
      std::fprintf(stderr, "%s 失败: %s\n", name,
                   image_codec_cuda_status_string(st));
      rc = 3;
    }
  };

  std::printf("size    : %ux%u x%u\n", a.width, a.height, a.channels);

  double mse = 0.0;
  show("mse", image_codec_cuda_metric_mse(codec, &a, &b, &mse));
  if (rc == 0) std::printf("mse     : %.6f\n", mse);

  double psnr = 0.0;
  show("psnr", image_codec_cuda_metric_psnr(codec, &a, &b, &psnr));
  if (rc == 0) std::printf("psnr    : %.4f dB\n", psnr);

  image_codec_cuda_ssim ssim{};
  show("ssim", image_codec_cuda_metric_ssim(codec, &a, &b, &ssim));
  if (rc == 0) {
    std::printf("ssim    : %.6f\n", ssim.ssim);
    std::printf("ms-ssim : %.6f (%d 级)\n", ssim.ms_ssim, ssim.levels);
  }

  image_codec_cuda_image_free(&a);
  image_codec_cuda_image_free(&b);
  image_codec_cuda_codec_destroy(codec);
  return rc;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fputs(kUsage, stderr);
    return 1;
  }
  const std::string cmd = argv[1];
  int device = 0;
  int iterations = 5;
  int fmt = IMAGE_CODEC_CUDA_FORMAT_UNKNOWN;
  image_codec_cuda_encode_params params{};
  params.jpeg_quality = 90;
  params.tiff_compression = 5;
  params.png_compression = 6;
  params.png_filter = -1;

  std::vector<const char *> positional;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s 缺少参数\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--device") {
      device = std::atoi(next("--device"));
    } else if (a == "--limit-mb") {
      gLimitBytes = static_cast<size_t>(std::atoll(next("--limit-mb"))) *
                    1024ull * 1024ull;
    } else if (a == "--iterations") {
      iterations = std::atoi(next("--iterations"));
    } else if (a == "--format") {
      fmt = formatFromName(next("--format"));
    } else if (a == "--quality") {
      params.jpeg_quality = std::atoi(next("--quality"));
    } else if (a == "--subsampling") {
      const std::string s = next("--subsampling");
      params.jpeg_subsampling = s == "444" ? 444 : (s == "422" ? 422 : 420);
    } else if (a == "--tiff-compression") {
      params.tiff_compression = std::atoi(next("--tiff-compression"));
    } else if (a == "--tile") {
      params.tiff_tile_size = std::atoi(next("--tile"));
    } else if (a == "--rows-per-strip") {
      params.tiff_rows_per_strip = std::atoi(next("--rows-per-strip"));
    } else if (a == "--png-filter") {
      params.png_filter = std::atoi(next("--png-filter"));
    } else if (a == "--png-compression") {
      params.png_compression = std::atoi(next("--png-compression"));
    } else {
      positional.push_back(argv[i]);
    }
  }

  if (cmd == "list-devices") {
    return cmdListDevices();
  }
  if (cmd == "info") {
    if (positional.empty()) {
      std::fputs(kUsage, stderr);
      return 1;
    }
    return cmdInfo(positional[0], device);
  }
  if (cmd == "decode") {
    if (positional.size() < 2) {
      std::fputs(kUsage, stderr);
      return 1;
    }
    return cmdDecode(positional[0], positional[1], device);
  }
  if (cmd == "encode") {
    if (positional.size() < 2) {
      std::fputs(kUsage, stderr);
      return 1;
    }
    if (fmt == IMAGE_CODEC_CUDA_FORMAT_UNKNOWN) {
      const std::string out = positional[1];
      const size_t dot = out.find_last_of('.');
      const std::string ext =
          dot == std::string::npos ? "" : out.substr(dot + 1);
      fmt = formatFromName(ext.c_str());
    }
    if (fmt == IMAGE_CODEC_CUDA_FORMAT_UNKNOWN) {
      std::fprintf(stderr, "无法确定输出格式，请用 --format\n");
      return 1;
    }
    return cmdEncode(positional[0], positional[1], fmt, device, params);
  }
  if (cmd == "mem") {
    return cmdMem(device);
  }
  if (cmd == "compare") {
    if (positional.size() < 2) {
      std::fputs(kUsage, stderr);
      return 1;
    }
    return cmdCompare(positional[0], positional[1], device);
  }
  if (cmd == "bench") {
    if (positional.empty()) {
      std::fputs(kUsage, stderr);
      return 1;
    }
    return cmdBench(positional[0], iterations, device);
  }
  std::fputs(kUsage, stderr);
  return 1;
}
