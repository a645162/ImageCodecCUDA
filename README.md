# ImageCodecCUDA

CUDA 后端图像编解码库：TIFF / PNG / JPEG 的 GPU 编解码、参考画质指标与硬件调参工具。

作为 [ImageCodecOffload](../ImageCodecOffload) 的三个 GPU 后端之一（OpenCL / CUDA / Vulkan），
对外提供稳定 C API（另附 C++ 包装），自身不负责后端选择——那由上层路由决定。

## 特性

| 格式 | 编码 | 解码 | 说明 |
| --- | --- | --- | --- |
| TIFF | GPU（LZW，strip/tile） | CPU | 6.0；Compression=1(none) / 5(LZW)；strip 与 tiled；1 / 2 / 3 / 4 通道 |
| PNG | GPU（行滤波，含 5 种滤波器自适应） | CPU（zlib） | 1 / 2 / 3 / 4 通道；滤波在 GPU，DEFLATE 在 CPU |
| JPEG | GPU（nvJPEG / NVJPG） | GPU（nvJPEG） | baseline，质量 1..100，4:4:4 / 4:2:2 / 4:2:0 |

- **通道保真**：1/2/3/4 通道各自原样往返（PNG color type 0/4/2/6、TIFF spp 1..4），
  绝不隐式升维或丢 Alpha。灰度 JPEG 走 `CSS_GRAY`（真 1 分量文件），
  而 JPEG 没有 Alpha 样本，2/4 通道会被明确拒绝。
- **画质指标**：MSE / PSNR / SSIM / MS-SSIM，GPU 并行，另有 CPU 参考实现逐项对拍。
- **显存可控**：可查询设备与句柄的显存用量，可设硬上限；超限给可读原因，绝不静默 OOM。
  资源不足与真失败用不同状态码（`OUT_OF_MEMORY` vs `ENCODE/DECODE_FAILED`），
  上层才知道该不该回退 CPU。
- **调参库与 CLI**：在自己硬件上扫描 tile/批大小/线程数/滤波策略，输出 Markdown 或 JSON。

## 构建

依赖 CUDA Toolkit（nvcc）与 vcpkg（`zlib` 端口）。CMake ≥ 3.24。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build -j
```

选项：`ICUDA_BUILD_CLI`（默认 ON）、`ICUDA_BUILD_EXAMPLES`（默认 ON）、`ICUDA_BUILD_TESTS`（默认 ON）。

### CUDA 架构

默认按 nvcc 版本生成，并与 `nvcc --list-gpu-arch` 取交集（CUDA 12.0~12.7 不认识 `sm_120`，
硬塞会 configure 失败）：

| nvcc | 默认架构 | 覆盖的显卡 |
| --- | --- | --- |
| ≥ 13 | `75;80;86;89;120` | Turing / Ampere / Ada / Blackwell（RTX 20~50 系） |
| < 13 | `61;75;80;86;89` | 额外含 Pascal（GTX 10 系） |

Volta（`sm_70/72`）任何情况下都不默认包含。显式指定：`-DCMAKE_CUDA_ARCHITECTURES=89`。

## CLI

```
icu list-devices
icu info <file>
icu decode <in> <out.bmp> [--device N]
icu encode <in.bmp> <out> [--format jpeg|tiff|png] [--quality N]
           [--subsampling 444|422|420] [--tiff-compression 1|5]
           [--tile N] [--rows-per-strip N] [--png-filter -1..4]
           [--png-compression 0-9] [--device N]
icu mem [--device N]                    # 设备显存 + 句柄记账/上限
icu compare <a.bmp> <b.bmp>             # MSE / PSNR / SSIM / MS-SSIM
icu bench <file> [--iterations N]

icu_tune --input <file.bmp> [--format tiff|png|jpeg] [选项]   # 参数扫描
```

通用选项：`--device N`（默认 0）、`--limit-mb N`（显存上限，0 = 不限制）。

示例：

```bash
icu encode big.bmp big.tiff --tile 256 --limit-mb 2048   # 大图 tiled LZW，限 2 GB 显存
icu compare orig.bmp decoded.bmp                          # 画质指标
icu_tune --input big.bmp --crop 0,0,4096,3072 --format tiff --tiles 128,256,512
```

## C API

```c
#include <ImageCodecCUDA/image_codec_cuda.h>

image_codec_cuda_codec *codec = NULL;
image_codec_cuda_codec_create(&codec, 0);
image_codec_cuda_set_memory_limit(codec, 2ull * 1024 * 1024 * 1024);

image_codec_cuda_image img = {0};
image_codec_cuda_decode(codec, IMAGE_CODEC_CUDA_FORMAT_UNKNOWN, data, size, &img);

double psnr = 0.0;
image_codec_cuda_metric_psnr(codec, &a, &b, &psnr);

image_codec_cuda_ssim ssim = {0};
image_codec_cuda_metric_ssim(codec, &a, &b, &ssim);   // ssim + ms_ssim + levels
```

所有入口返回 `image_codec_cuda_status`（数值与 ImageCodecOpenCL 对齐，上层可以只用一套 switch）。

## 文档

- [`docs/metrics.md`](docs/metrics.md) —— 画质指标的精确定义、跨后端一致性要求。
- [`docs/tune.md`](docs/tune.md) —— 调参原理、CLI 用法与实测样例。
- [`docs/architecture.md`](docs/architecture.md) —— 模块划分、显存策略与关键取舍。

## 已知限制

- **PNG DEFLATE 在 CPU**：GPU 上做 Huffman/滑动窗口压缩的收益不足以抵消调度与同步开销，
  这里只把最贵的「逐像素行滤波」放到 GPU（含自适应选择），这与 TIFF LZW 只把编码放 GPU 是同一思路。
- **LZW 解码在 CPU**：解码是串行字典型状态机，GPU 并行化需要多轮前缀扫描，实测不占优。
- **JPEG 无渐进式**：nvJPEG 只处理 baseline/渐进式解码；编码固定 baseline。
- 单进程内不使用隐式默认 context，每个 codec 句柄显式持有一个 stream。
