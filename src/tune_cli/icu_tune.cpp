/*
 * File: icu_tune.cpp
 * Brief: Icu tune.
 * Author: Haomin Kong
 * Email: a645162@163.com
 * Created: 2026-10-31
 */

// icu_tune.cpp - 参数扫描 CLI（ImageCodecCUDA::Tune 的薄壳）。
//
// 典型用法：
//   icu_tune --input big.bmp --format tiff                       # 跑默认候选集
//   icu_tune --input big.bmp --crop 0,0,4096,4096 --format png   # 大图只调一块
//   icu_tune --input big.bmp --format tiff --tiles 128,256,512 --threads 128,256
//   icu_tune --input big.bmp --format tiff --limit-mb 2048 --json tiff.json
//
// 命令行给的扫描维度会与默认候选集的其余维度做笛卡尔积；一个都不给就用默认集。
#include <ImageCodecCUDA/image_codec_cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tune.h"

using image_codec::cuda::ImageBuffer;
using image_codec::cuda::tune::Candidate;
using image_codec::cuda::tune::Format;
using image_codec::cuda::tune::Options;

namespace {

const char *kUsage =
    "用法: icu_tune --input <file.bmp> [选项]\n"
    "\n"
    "  --format tiff|png|jpeg   目标格式（默认 tiff）\n"
    "  --device N               CUDA 设备号（默认 0）\n"
    "  --iters N                每个候选计时次数（默认 3，取最小值）\n"
    "  --warmup N               预热次数（默认 1）\n"
    "  --limit-mb N             显存上限（MB，默认不限制）\n"
    "  --crop x,y,w,h           只取 BMP 的一块（大图调参强烈建议，读盘也省）\n"
    "  --no-verify              跳过往返校验（不建议：快但错的组合没有意义）\n"
    "  --stop-on-error          某个候选失败就停止\n"
    "  --json <path>            结果写成 JSON\n"
    "  --markdown <path>        结果写成 Markdown 表格\n"
    "\n"
    "扫描维度（不给则用默认候选集）:\n"
    "  --tiles 0,128,256,512,1024   TIFF tile 边长（0 = 用 strip）\n"
    "  --rows 8,32,64,128           TIFF strip 行数（tiles=0 时生效）\n"
    "  --lzw-threads 128,256,512    TIFF LZW 每 block 线程数\n"
    "  --batches 0,64,256           TIFF 每批上传的单元数（0 = "
    "按显存预算自动）\n"
    "  --filters -1,0,1,2,3,4       PNG 行滤波（-1 = 自适应）\n"
    "  --levels 1,6,9               PNG zlib 等级\n"
    "  --png-threads 128,256,512    PNG 滤波 kernel 线程数\n"
    "  --qualities 90,95            JPEG 质量\n"
    "  --subsampling 444,420,422    JPEG 子采样\n";

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    const size_t pos = s.find(',', start);
    const std::string part = s.substr(
        start, pos == std::string::npos ? std::string::npos : pos - start);
    if (!part.empty()) out.push_back(part);
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return out;
}

std::vector<int> toInts(const std::string &s) {
  std::vector<int> out;
  for (const auto &p : split(s)) out.push_back(std::atoi(p.c_str()));
  return out;
}

bool readFile(const char *path, std::vector<uint8_t> *out) {
  FILE *f = std::fopen(path, "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len <= 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(len));
  const size_t got = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);
  return got == out->size();
}

void writeText(const char *path, const std::string &text) {
  FILE *f = std::fopen(path, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "无法写入: %s\n", path);
    return;
  }
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}

// 命令行维度 -> 候选集（笛卡尔积）。
std::vector<Candidate> buildCandidates(
    Format format, const std::vector<int> &tiles, const std::vector<int> &rows,
    const std::vector<int> &lzw_threads, const std::vector<int> &batches,
    const std::vector<int> &filters, const std::vector<int> &levels,
    const std::vector<int> &png_threads, const std::vector<int> &qualities,
    const std::vector<int> &subs) {
  const bool any_tiff = !tiles.empty() || !rows.empty() ||
                        !lzw_threads.empty() || !batches.empty();
  const bool any_png =
      !filters.empty() || !levels.empty() || !png_threads.empty();
  const bool any_jpeg = !qualities.empty() || !subs.empty();

  if (format == Format::Tiff) {
    if (!any_tiff) return image_codec::cuda::tune::defaultCandidates(format);
    const std::vector<int> t = tiles.empty() ? std::vector<int>{0, 256} : tiles;
    const std::vector<int> r = rows.empty() ? std::vector<int>{0} : rows;
    const std::vector<int> th =
        lzw_threads.empty() ? std::vector<int>{0} : lzw_threads;
    const std::vector<int> b = batches.empty() ? std::vector<int>{0} : batches;
    std::vector<Candidate> out;
    for (int tv : t) {
      for (int rv : r) {
        if (tv != 0 && rv != 0) continue; // tile 与 strip 行数互斥
        for (int thv : th) {
          for (int bv : b) {
            Candidate c;
            c.tiff_tile_size = tv;
            c.tiff_rows_per_strip = tv == 0 ? rv : 0;
            c.tiff_lzw_block_threads = thv;
            c.tiff_lzw_unit_batch = bv;
            out.push_back(c);
          }
        }
      }
    }
    return out;
  }
  if (format == Format::Png) {
    if (!any_png) return image_codec::cuda::tune::defaultCandidates(format);
    const std::vector<int> f = filters.empty() ? std::vector<int>{-1} : filters;
    const std::vector<int> l = levels.empty() ? std::vector<int>{6} : levels;
    const std::vector<int> th =
        png_threads.empty() ? std::vector<int>{0} : png_threads;
    std::vector<Candidate> out;
    for (int fv : f) {
      for (int lv : l) {
        for (int thv : th) {
          Candidate c;
          c.png_filter = fv;
          c.png_compression = lv;
          c.png_block_threads = thv;
          out.push_back(c);
        }
      }
    }
    return out;
  }
  if (!any_jpeg) return image_codec::cuda::tune::defaultCandidates(format);
  const std::vector<int> q =
      qualities.empty() ? std::vector<int>{90} : qualities;
  const std::vector<int> s = subs.empty() ? std::vector<int>{0} : subs;
  std::vector<Candidate> out;
  for (int qv : q) {
    for (int sv : s) {
      Candidate c;
      c.jpeg_quality = qv;
      c.jpeg_subsampling = sv;
      out.push_back(c);
    }
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::string input;
  std::string json_path;
  std::string md_path;
  Format format = Format::Tiff;
  Options options;
  int crop_x = 0;
  int crop_y = 0;
  int crop_w = 0;
  int crop_h = 0;
  size_t limit_mb = 0;
  std::vector<int> tiles, rows, lzw_threads, batches, filters, levels,
      png_threads, qualities, subs;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s 缺少参数\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--input") {
      input = next("--input");
    } else if (a == "--format") {
      const std::string f = next("--format");
      format = f == "png"
                   ? Format::Png
                   : (f == "jpeg" || f == "jpg" ? Format::Jpeg : Format::Tiff);
    } else if (a == "--device") {
      options.device_index = std::atoi(next("--device"));
    } else if (a == "--iters") {
      options.iterations = std::atoi(next("--iters"));
    } else if (a == "--warmup") {
      options.warmup = std::atoi(next("--warmup"));
    } else if (a == "--limit-mb") {
      limit_mb = static_cast<size_t>(std::atoll(next("--limit-mb")));
    } else if (a == "--no-verify") {
      options.verify = false;
    } else if (a == "--stop-on-error") {
      options.stop_on_error = true;
    } else if (a == "--json") {
      json_path = next("--json");
    } else if (a == "--markdown") {
      md_path = next("--markdown");
    } else if (a == "--crop") {
      const auto v = toInts(next("--crop"));
      if (v.size() == 4) {
        crop_x = v[0];
        crop_y = v[1];
        crop_w = v[2];
        crop_h = v[3];
      }
    } else if (a == "--tiles") {
      tiles = toInts(next("--tiles"));
    } else if (a == "--rows") {
      rows = toInts(next("--rows"));
    } else if (a == "--lzw-threads") {
      lzw_threads = toInts(next("--lzw-threads"));
    } else if (a == "--batches") {
      batches = toInts(next("--batches"));
    } else if (a == "--filters") {
      filters = toInts(next("--filters"));
    } else if (a == "--levels") {
      levels = toInts(next("--levels"));
    } else if (a == "--png-threads") {
      png_threads = toInts(next("--png-threads"));
    } else if (a == "--qualities") {
      qualities = toInts(next("--qualities"));
    } else if (a == "--subsampling") {
      subs = toInts(next("--subsampling"));
    } else if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    } else {
      std::fprintf(stderr, "未知参数: %s\n", a.c_str());
      std::fputs(kUsage, stderr);
      return 1;
    }
  }

  if (input.empty()) {
    std::fputs(kUsage, stderr);
    return 1;
  }
  options.memory_limit = limit_mb * 1024ull * 1024ull;

  // 先把显存状况打出来：调参时「显存不够导致回退」是最容易被误读的噪声。
  image_codec_cuda_memory_info mem{};
  if (image_codec_cuda_device_memory(options.device_index, &mem) ==
      IMAGE_CODEC_CUDA_OK) {
    std::printf("显存: 总 %zu MiB, 空闲 %zu MiB%s\n", mem.total_bytes >> 20,
                mem.free_bytes >> 20,
                options.memory_limit > 0 ? " (已设上限)" : "");
  }

  ImageBuffer img;
  std::string err;
  const double t0 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
  if (!image_codec::cuda::tune::loadBmpRegion(input.c_str(), crop_x, crop_y,
                                              crop_w, crop_h, &img, &err)) {
    std::fprintf(stderr, "读图失败: %s\n", err.c_str());
    return 2;
  }
  const double t1 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
  std::printf("输入: %dx%d x%d (%zu MiB), 读取 %.2f s\n", img.width, img.height,
              img.channels, img.bytes() >> 20, t1 - t0);

  const auto candidates =
      buildCandidates(format, tiles, rows, lzw_threads, batches, filters,
                      levels, png_threads, qualities, subs);
  std::printf("候选数: %zu（每个 %d 次计时 + %d 次预热）\n", candidates.size(),
              options.iterations, options.warmup);

  const auto results =
      image_codec::cuda::tune::sweep(img, format, candidates, options);
  const std::string md = image_codec::cuda::tune::toMarkdown(results, format);
  std::fputs(md.c_str(), stdout);
  if (!md_path.empty()) writeText(md_path.c_str(), md);
  if (!json_path.empty()) {
    writeText(json_path.c_str(),
              image_codec::cuda::tune::toJson(results, format));
  }

  int failures = 0;
  for (const auto &r : results) {
    if (!r.ok) ++failures;
  }
  return failures == 0 ? 0 : 3;
}
