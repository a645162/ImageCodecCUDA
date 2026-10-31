# 架构

本库是 ImageCodecOffload 的 CUDA 后端。整体分工是：

```
        ImageCodecOffload（路由层：选后端、选格式、给 CPU 兜底）
                 │  C API: image_codec_cuda_*
                 ▼
        Cuda.Image.Codec.Api          ← 参数校验 / 格式分派 / C 结构体 <-> ImageBuffer
        ├─ Cuda.Image.Codec.Tiff      ← TIFF 容器 + GPU LZW 编码
        ├─ Cuda.Image.Codec.Png       ← PNG 容器 + GPU 行滤波
        ├─ Nvidia.Image.Codec.NvJpeg  ← nvJPEG / NVJPG
        ├─ Cuda.Image.Codec.Metrics   ← MSE / PSNR / SSIM / MS-SSIM
        ├─ Cuda.Image.Codec.Tune      ← 参数扫描（库，不是脚本）
        └─ Cuda.Image.Codec.Common    ← ImageBuffer / LZW / BMP / 容器读写（CPU）
                 │
                 ▼
        Cuda.Image.Codec.Core         ← Device / Stream / DeviceBuffer / MemoryBudget
```

## 为什么这样切

**Core 不知道任何格式。** `Device`、`DeviceBuffer`、`MemoryBudget` 是唯一与
`cudaMalloc`/`cudaMemcpy` 打交道的地方。格式模块拿到的是「设备上的一块缓冲」，
不自己调 CUDA 分配——否则显存上限就只能靠每个模块各自记得检查，迟早漏。

**Common 全部是 CPU 代码。** TIFF/PNG 容器解析、LZW 解码、DEFLATE、BMP 读写都留在这里。
这不是「还没 GPU 化」，是刻意的取舍，理由见下面「只把最贵的一步放 GPU」。

**Api 不写格式逻辑。** 它只做三件事：参数校验、按格式分派、`image_codec_cuda_image`
与内部 `ImageBuffer` 之间的转换。一旦它开始长格式逻辑，就成了第二个 codec。

**Tune 不进 C API。** 调参需要精确计时，而 C API 的 `image` 结构体没有 stride/所有权概念，
每次调用必然复制整幅 host 图像（1.9 GB 的图光拷贝就淹没了被测时间）。因此 Tune 直接链
内部模块入口，`icu_tune` 是它的薄壳。

**Metrics 独立成一个模块**，而不是塞进编解码里：指标要被 Tune、CLI、上层程序共用，
且三个后端必须给出同一个数字（见 [`metrics.md`](metrics.md)）。

## 显存策略

大图是硬约束：`Merge.bmp` 是 32000×19656×3 = 1.887 GB，两张图同时驻留设备就是 3.8 GB。

三条规则：

1. **不整幅上传。** 指标按行 band 流式处理；TIFF LZW 按 tile 批次上传；分块大小由调用方
   或显存预算决定，与图像总大小无关。
2. **所有设备分配走 `MemoryBudget`。** `DeviceBuffer::ensure()` 在真正 `cudaMalloc` 之前
   先问预算；超出上限时给出**可读原因**（需要多少、上限多少、已用多少），而不是等到驱动
   抛 `out of memory` 或整机进 swap。
3. **分配前先问驱动。** 大块分配前用 `cudaMemGetInfo` 提前判断，避免「预算是够的但设备
   已被别的进程吃掉」这种情形最后变成一个看不懂的 CUDA 错误码。

```c
image_codec_cuda_memory_info mi;
image_codec_cuda_device_memory(0, &mi);      // 设备视角：总 / 空闲
image_codec_cuda_get_memory_info(codec, &mi); // 句柄视角：上限 / 已用 / 峰值
image_codec_cuda_set_memory_limit(codec, 2ull << 30);
```

句柄与设备是两个视角，必须分开：设备空闲显存受其它进程影响，瞬时可查但不代表本句柄
能用到；句柄的 `budget/used/peak` 是本库自己记账的结果，可以稳定用于决策。

## 通道保真

**1 通道进必须 1 通道出，3 通道进必须 3 通道出。** 这条不是洁癖：上层用通道数判断
「这张图是不是灰度」「体积为什么变成三倍」，一旦某个后端把灰度悄悄展开成 RGB，上层的
判断全部错位，而肉眼几乎看不出来。三个格式各有各的坑：

| 格式 | 坑 | 做法 |
| --- | --- | --- |
| JPEG | nvJPEG 的编码输入只有 3 通道枚举（`NVJPEG_INPUT_RGB/BGR/RGBI/BGRI`），没有单通道入口；只把输入凑成 3 通道而采样因子仍是 4:2:0，输出的就是三分量 JPEG | 宿主侧复制成 R=G=B，同时把采样因子强制成 `NVJPEG_CSS_GRAY`；输出 `ncomp==1`。解码侧按 `ncomp==1 \|\| css==GRAY` 走 `NVJPEG_OUTPUT_Y`，返回 1 通道 |
| PNG | 解码走 zlib + CPU 反滤波，通道数来自 IHDR 的 color type，天然保真 | 不解码成固定 RGB；1/3/4 通道各自成路 |
| TIFF | `SamplesPerPixel` 与 `Photometric` 必须一起写对，写错只会在别的解码器里显形 | 按 `spp` 写 `Photometric=RGB(2)` / `BlackIsZero(1)` |

`codec_roundtrip_test` 的 `testJpegGray` 专门锁这条：灰度 JPEG 解码结果必须是 1 通道，
且 PSNR > 32 dB。

## 只把最贵的一步放 GPU

三个格式都是「GPU 做其中一段，其余留 CPU」，这不是半成品：

| 格式 | 放 GPU 的部分 | 留 CPU 的部分 | 理由 |
| --- | --- | --- | --- |
| TIFF | LZW 编码（一线程一 strip/tile） | LZW 解码 | 编码是纯函数式的字典构建，可完全并行；解码是串行状态机，GPU 化需要多轮前缀扫描，实测不占优 |
| PNG | 逐像素行滤波（含 5 种滤波器自适应选择） | DEFLATE | 滤波是逐像素独立运算；DEFLATE 的滑动窗口 + Huffman 是串行的，调度与同步开销超过收益 |
| JPEG | 全部（nvJPEG/NVJPG） | 无 | NVIDIA 提供了完整的硬件实现，没有理由自己写 DCT |

判断标准只有一条：**这段代码是否逐元素独立**。是就上 GPU，否则留在 CPU 并接受它。
把串行压缩算法硬塞进 GPU，最后得到的是一个更难维护、还更慢的实现。

## 确定性

同一个输入、同一组参数，编码结果必须逐字节一致。这条看起来是常识，但在 CUDA 上很容易
被破坏——本库曾经就因为 `Device` 用 `cudaStreamNonBlocking`、而 `DeviceBuffer::upload`
走 legacy default stream，两者没有隐式顺序，导致 kernel 偶尔读到没落地的上传数据，
同一张 PNG 两次编码的字节数会跳变。修法是参数化 stream + `cudaMemcpyAsync` + 同流同步，
并在 `codec_roundtrip_test` 里加了 `testPngDeterminism` 锁住。

自适应滤波这类「按统计量选分支」的路径尤其要注意：它读到的任何一点脏数据都会改变选中
的滤波器，进而改变整行输出。

## 错误模型

- **C API 只返回状态码**（数值与 ImageCodecOpenCL 对齐，上层可以只用一套 switch）。
- 需要具体原因（例如「显存预算不足：需要 25 MB，上限 8 MB」）时走内部 C++ 入口，接口带
  `std::string* err`。
- 不允许把失败吞掉降级成「另一个能跑的实现」：某后端不支持的算子要返回显式错误并附原因，
  由上层决定是否回退 CPU。

### 失败分类（决定上层该不该回退）

状态码本身不够用：「这台设备的资源不够」换 CPU 还有救，「这张图真的编坏了」换后端只会
把 bug 藏起来。两者必须给出不同状态码，否则上层只能猜。分类集中在
`Cuda.Image.Codec.Api` 的 `classifyFailure()`：

| 情形 | 状态码 | 上层行为 |
| --- | --- | --- |
| 显存预算 / 设备空闲显存不足；nvJPEG 内部 `allocator failure` | `OUT_OF_MEMORY` | 可回退（CPU 用内存顶上） |
| nvJPEG `not supported` / `not available`（渐进式、16 bit、缺组件） | `UNSUPPORTED_FORMAT` | 可回退（别的解码器可能能读） |
| 其余 | `ENCODE_FAILED` / `DECODE_FAILED` | 真失败，不允许被别的后端悄悄救回来 |

判断依据不是匹配错误字符串，而是 `MemoryBudget::exhausted()`——它在 `reserve()` 里就
已知道「是预算/设备显存不够」。每次操作前 `takeExhausted()` 清掉上一轮结论，避免这一轮
的失败被上一轮的结论污染。

一个真实的例子：Merge.bmp（32000×19656，629 Mpix）按质量 92 编码会选 4:4:4，nvJPEG 在
这个尺寸上返回 `allocator failure`（显存远未用满，是它自身的缓冲限制），4:2:0 / 4:2:2
则正常。这必须表现为 `OUT_OF_MEMORY` 让上层回退 CPU，而不是「编码失败」。**不要**为此
静默降采样——那会悄悄改写调用方请求的画质语义。
