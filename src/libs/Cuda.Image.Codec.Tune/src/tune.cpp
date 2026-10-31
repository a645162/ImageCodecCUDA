/*
 * File: tune.cpp
 * Brief: Tune.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// tune.cpp - 参数扫描实现。
#include "tune.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

#include "bmp_io.h"
#include "jpeg_codec.h"
#include "metrics.h"
#include "png_codec.h"
#include "tiff_codec.h"

namespace image_codec {
namespace cuda {
namespace tune {

namespace {

double nowMs() {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string makeLabel(const Candidate &c, Format format) {
  if (!c.label.empty()) return c.label;
  std::ostringstream os;
  switch (format) {
  case Format::Tiff:
    os << (c.tiff_tile_size > 0
               ? "tile=" + std::to_string(c.tiff_tile_size)
               : "strip rows=" + std::to_string(c.tiff_rows_per_strip))
       << " lzw_threads=" << c.tiff_lzw_block_threads
       << " batch=" << c.tiff_lzw_unit_batch << " comp=" << c.tiff_compression;
    break;
  case Format::Png:
    os << "filter=" << c.png_filter << " level=" << c.png_compression
       << " threads=" << c.png_block_threads;
    break;
  case Format::Jpeg:
    os << "q=" << c.jpeg_quality << " sub=" << c.jpeg_subsampling;
    break;
  }
  return os.str();
}

} // namespace

const char *formatName(Format format) {
  switch (format) {
  case Format::Tiff:
    return "tiff";
  case Format::Png:
    return "png";
  case Format::Jpeg:
    return "jpeg";
  }
  return "unknown";
}

std::vector<Candidate> defaultCandidates(Format format) {
  std::vector<Candidate> out;
  if (format == Format::Tiff) {
    const int tiles[] = {0, 128, 256, 512, 1024};
    const int rows[] = {8, 32, 128};
    const int threads[] = {128, 256};
    for (int t : tiles) {
      for (int th : threads) {
        Candidate c;
        c.tiff_compression = 5;
        c.tiff_lzw_block_threads = th;
        if (t == 0) {
          for (int r : rows) {
            Candidate s = c;
            s.tiff_tile_size = 0;
            s.tiff_rows_per_strip = r;
            s.label = makeLabel(s, format);
            out.push_back(s);
          }
        } else {
          c.tiff_tile_size = t;
          c.label = makeLabel(c, format);
          out.push_back(c);
        }
      }
    }
    return out;
  }
  if (format == Format::Png) {
    const int filters[] = {-1, 0, 1, 2, 3, 4};
    const int levels[] = {1, 6, 9};
    const int threads[] = {128, 256, 512};
    for (int f : filters) {
      for (int l : levels) {
        for (int th : threads) {
          Candidate c;
          c.png_filter = f;
          c.png_compression = l;
          c.png_block_threads = th;
          c.label = makeLabel(c, format);
          out.push_back(c);
        }
      }
    }
    return out;
  }
  const int qualities[] = {90, 95};
  const int subs[] = {444, 420, 422};
  for (int q : qualities) {
    for (int s : subs) {
      Candidate c;
      c.jpeg_quality = q;
      c.jpeg_subsampling = s;
      c.label = makeLabel(c, format);
      out.push_back(c);
    }
  }
  return out;
}

Result run(const ImageBuffer &img, Format format, const Candidate &candidate,
           const Options &options) {
  Result r;
  r.label = makeLabel(candidate, format);
  MemoryBudget budget;
  budget.bindDevice(options.device_index);
  budget.setLimit(options.memory_limit);

  std::vector<double> samples;
  const int warmup = std::max(0, options.warmup);
  const int iterations = std::max(1, options.iterations);

  for (int i = 0; i < warmup + iterations; ++i) {
    const bool timed = i >= warmup;
    std::vector<uint8_t> encoded;
    bool gpu_used = false;
    std::string err;
    const double t0 = nowMs();
    bool ok = false;

    switch (format) {
    case Format::Tiff: {
      tiff::TiffEncodeOptions opt;
      opt.compression = candidate.tiff_compression;
      opt.rows_per_strip = candidate.tiff_rows_per_strip;
      opt.tile_size = candidate.tiff_tile_size;
      opt.lzw_block_threads = candidate.tiff_lzw_block_threads;
      opt.lzw_unit_batch = candidate.tiff_lzw_unit_batch;
      opt.budget = &budget;
      ok = tiff::encodeTiff(img, opt, &encoded, &gpu_used, &err);
      break;
    }
    case Format::Png: {
      png::PngEncodeOptions opt;
      opt.filter = candidate.png_filter;
      opt.compression_level = candidate.png_compression;
      opt.block_threads = candidate.png_block_threads;
      opt.budget = &budget;
      ok = png::encodePng(img, opt, &encoded, &gpu_used, &err);
      break;
    }
    case Format::Jpeg: {
      ok = jpeg::encode(options.device_index, img, candidate.jpeg_quality,
                        candidate.jpeg_subsampling, &encoded, &err, &budget);
      break;
    }
    }
    const double elapsed = nowMs() - t0;

    if (!ok) {
      r.ok = false;
      r.error = err.empty() ? "encode failed" : err;
      return r;
    }
    if (timed) samples.push_back(elapsed);
    r.output_bytes = encoded.size();
    r.gpu_used = gpu_used;

    // 只用最后一次计时样本做校验（省时间，正确性不依赖迭代次数）。
    if (options.verify && timed && i == warmup + iterations - 1) {
      ImageBuffer back;
      std::string derr;
      bool decoded = false;
      switch (format) {
      case Format::Tiff:
        decoded =
            tiff::decodeTiff(encoded.data(), encoded.size(), &back, &derr);
        break;
      case Format::Png:
        decoded = png::decodePng(encoded.data(), encoded.size(), &back, &derr);
        break;
      case Format::Jpeg:
        decoded = jpeg::decode(options.device_index, encoded.data(),
                               encoded.size(), &back, &derr, &budget);
        break;
      }
      if (!decoded) {
        r.ok = false;
        r.error = "decode failed: " + derr;
        return r;
      }
      if (back.width != img.width || back.height != img.height ||
          back.channels != img.channels) {
        r.ok = false;
        r.error = "roundtrip dimension mismatch";
        return r;
      }
      if (format == Format::Jpeg) {
        // 有损格式：只校验尺寸、通道与画质下限（PSNR / SSIM / MS-SSIM 走统一的
        // 指标模块，保证「调参工具认为达标」与产品侧判断用的是同一套定义）。
        metrics::Options mopt;
        mopt.device_index = options.device_index;
        mopt.memory_limit = options.memory_limit;

        double jpeg_psnr = 0.0;
        std::string merr;
        if (!metrics::psnr(img, back, mopt, &jpeg_psnr, &merr)) {
          r.ok = false;
          r.error = "metrics failed: " + merr;
          return r;
        }

        metrics::SsimResult jpeg_ssim;
        if (!metrics::ssim(img, back, mopt, &jpeg_ssim, &merr)) {
          r.ok = false;
          r.error = "metrics failed: " + merr;
          return r;
        }

        if (jpeg_psnr < 30.0 || jpeg_ssim.ssim < 0.9) {
          r.ok = false;
          r.error = "jpeg quality too low (psnr " + std::to_string(jpeg_psnr) +
                    " dB, ssim " + std::to_string(jpeg_ssim.ssim) + ")";
          return r;
        }

        r.psnr = jpeg_psnr;
        r.ssim = jpeg_ssim.ssim;
        r.ms_ssim = jpeg_ssim.ms_ssim;
      } else if (back.data != img.data) {
        r.ok = false;
        r.error = "roundtrip not bit-exact";
        return r;
      }
    }
  }

  r.ok = true;
  r.best_ms = *std::min_element(samples.begin(), samples.end());
  r.median_ms = median(samples);
  r.peak_vram = budget.peak();
  const double bytes = static_cast<double>(img.bytes());
  r.input_mb_per_s = r.best_ms > 0.0 ? bytes / 1e6 / (r.best_ms / 1000.0) : 0.0;
  return r;
}

std::vector<Result> sweep(const ImageBuffer &img, Format format,
                          const std::vector<Candidate> &candidates,
                          const Options &options) {
  std::vector<Result> results;
  results.reserve(candidates.size());
  for (const auto &c : candidates) {
    Result r = run(img, format, c, options);
    const bool failed = !r.ok;
    if (failed && options.stop_on_error) {
      results.push_back(std::move(r));
      break;
    }
    results.push_back(std::move(r));
  }
  std::stable_sort(results.begin(), results.end(),
                   [](const Result &a, const Result &b) {
                     if (a.ok != b.ok) return a.ok;
                     return a.best_ms < b.best_ms;
                   });
  return results;
}

std::string toMarkdown(const std::vector<Result> &results, Format format) {
  std::ostringstream os;
  os << "| # | 参数 | 最优耗时 (ms) | 中位数 (ms) | 吞吐 (MB/s) | 输出 (KB) "
        "| 显存峰值 (MB) | 画质 | GPU | 校验 |\n";
  os << "|---|---|---|---|---|---|---|---|---|---|\n";
  int rank = 0;
  for (const auto &r : results) {
    ++rank;
    os << "| " << rank << " | " << r.label << " | ";
    if (r.ok) {
      os << r.best_ms << " | " << r.median_ms << " | " << r.input_mb_per_s
         << " | " << (r.output_bytes / 1024) << " | "
         << (r.peak_vram / (1024.0 * 1024.0)) << " | ";
      if (r.psnr > 0.0) {
        os << "psnr " << r.psnr << "/ssim " << r.ssim << "/ms-ssim "
           << r.ms_ssim;
      } else {
        os << "lossless";
      }
      os << " | " << (r.gpu_used ? "yes" : "no") << " | ok |";
    } else {
      os << "- | - | - | " << (r.output_bytes / 1024) << " | - | - | - | "
         << r.error << " |";
    }
    os << "\n";
  }
  os << "\n格式: " << formatName(format) << "\n";
  if (!results.empty() && results.front().ok) {
    os << "\n推荐（按最优耗时）: `" << results.front().label << "`\n";
  }
  return os.str();
}

std::string toJson(const std::vector<Result> &results, Format format) {
  std::ostringstream os;
  os << "{\n  \"format\": \"" << formatName(format) << "\",\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    os << "    {\"label\": \"" << r.label
       << "\", \"ok\": " << (r.ok ? "true" : "false")
       << ", \"best_ms\": " << r.best_ms << ", \"median_ms\": " << r.median_ms
       << ", \"input_mb_per_s\": " << r.input_mb_per_s
       << ", \"output_bytes\": " << r.output_bytes
       << ", \"peak_vram\": " << r.peak_vram << ", \"psnr\": " << r.psnr
       << ", \"ssim\": " << r.ssim << ", \"ms_ssim\": " << r.ms_ssim
       << ", \"gpu_used\": " << (r.gpu_used ? "true" : "false")
       << ", \"error\": \"" << r.error << "\"}";
    if (i + 1 != results.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n}\n";
  return os.str();
}

bool loadBmpRegion(const char *path, int x, int y, int crop_w, int crop_h,
                   ImageBuffer *out, std::string *err) {
  if (path == nullptr || out == nullptr) {
    if (err != nullptr) *err = "loadBmpRegion: null argument";
    return false;
  }
  FILE *f = std::fopen(path, "rb");
  if (f == nullptr) {
    if (err != nullptr)
      *err = std::string("loadBmpRegion: cannot open ") + path;
    return false;
  }
  struct Closer {
    FILE *f;
    ~Closer() {
      if (f != nullptr) std::fclose(f);
    }
  } closer{f};

  uint8_t file_header[14];
  uint8_t info_header[40];
  if (std::fread(file_header, 1, sizeof(file_header), f) !=
          sizeof(file_header) ||
      std::fread(info_header, 1, sizeof(info_header), f) !=
          sizeof(info_header)) {
    if (err != nullptr) *err = "loadBmpRegion: truncated header";
    return false;
  }
  if (file_header[0] != 'B' || file_header[1] != 'M') {
    if (err != nullptr) *err = "loadBmpRegion: not a BMP";
    return false;
  }
  const uint32_t data_offset = static_cast<uint32_t>(file_header[10]) |
                               (static_cast<uint32_t>(file_header[11]) << 8) |
                               (static_cast<uint32_t>(file_header[12]) << 16) |
                               (static_cast<uint32_t>(file_header[13]) << 24);
  const int32_t raw_w =
      static_cast<int32_t>(static_cast<uint32_t>(info_header[4]) |
                           (static_cast<uint32_t>(info_header[5]) << 8) |
                           (static_cast<uint32_t>(info_header[6]) << 16) |
                           (static_cast<uint32_t>(info_header[7]) << 24));
  const int32_t raw_h =
      static_cast<int32_t>(static_cast<uint32_t>(info_header[8]) |
                           (static_cast<uint32_t>(info_header[9]) << 8) |
                           (static_cast<uint32_t>(info_header[10]) << 16) |
                           (static_cast<uint32_t>(info_header[11]) << 24));
  const uint16_t bpp =
      static_cast<uint16_t>(static_cast<uint32_t>(info_header[14]) |
                            (static_cast<uint32_t>(info_header[15]) << 8));
  const uint32_t compression = static_cast<uint32_t>(info_header[16]) |
                               (static_cast<uint32_t>(info_header[17]) << 8) |
                               (static_cast<uint32_t>(info_header[18]) << 16) |
                               (static_cast<uint32_t>(info_header[19]) << 24);
  if (raw_w <= 0 || raw_h == 0 || bpp != 24 || compression != 0) {
    if (err != nullptr) {
      *err = "loadBmpRegion: only uncompressed 24-bit BMP is supported";
    }
    return false;
  }
  const bool bottom_up = raw_h > 0;
  const int img_w = raw_w;
  const int img_h = raw_h > 0 ? raw_h : -raw_h;
  const int src_stride = ((img_w * 3 + 3) / 4) * 4;

  const int x0 = std::clamp(x, 0, img_w);
  const int y0 = std::clamp(y, 0, img_h);
  int w = crop_w > 0 ? crop_w : img_w - x0;
  int h = crop_h > 0 ? crop_h : img_h - y0;
  w = std::min(w, img_w - x0);
  h = std::min(h, img_h - y0);
  if (w <= 0 || h <= 0) {
    if (err != nullptr) *err = "loadBmpRegion: empty region";
    return false;
  }
  if (!out->allocate(w, h, 3)) {
    if (err != nullptr) *err = "loadBmpRegion: allocation failed";
    return false;
  }

  std::vector<uint8_t> row(static_cast<size_t>(src_stride));
  for (int ry = 0; ry < h; ++ry) {
    // BMP 行序与 top-down 相反：正高度表示文件里最后一行是图像顶部。
    const int img_row = y0 + ry;
    const int file_row = bottom_up ? (img_h - 1 - img_row) : img_row;
    const long offset = static_cast<long>(data_offset) +
                        static_cast<long>(file_row) * src_stride +
                        static_cast<long>(x0) * 3;
    if (std::fseek(f, offset, SEEK_SET) != 0 ||
        std::fread(row.data(), 1, row.size(), f) != row.size()) {
      if (err != nullptr) *err = "loadBmpRegion: read past end of file";
      return false;
    }
    uint8_t *dst = out->data.data() + static_cast<size_t>(ry) * out->stride();
    for (int rx = 0; rx < w; ++rx) {
      dst[rx * 3 + 0] = row[static_cast<size_t>(rx) * 3 + 2]; // B -> R
      dst[rx * 3 + 1] = row[static_cast<size_t>(rx) * 3 + 1]; // G -> G
      dst[rx * 3 + 2] = row[static_cast<size_t>(rx) * 3 + 0]; // R -> B
    }
  }
  return true;
}

} // namespace tune
} // namespace cuda
} // namespace image_codec
