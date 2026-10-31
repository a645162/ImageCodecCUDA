# 参数调优（`ImageCodecCUDA::Tune` 与 `icu_tune`）

## 为什么需要它

编码参数（TIFF tile 边长、每批上送单元数、LZW 每 block 线程数、PNG 滤波策略与 zlib 等级、
JPEG 质量与子采样）的最优点与 **具体 GPU、显存容量、图像尺寸** 强相关。别人测出来的数字
在你的机器上往往不成立——所以调参必须是「在本机跑一遍」，而不是查一张表。

`Cuda.Image.Codec.Tune` 把这件事做成可复用的 C++ 库，`icu_tune` 只是薄壳；上层程序
（ImageViewer 的基准页、CI 的性能门限）可以直接链接这个库。

## 测量约定

| 约定 | 原因 |
| --- | --- |
| 直接调内部模块入口，不经 C API | C API 为 ABI 稳定会复制一份 host 缓冲；1.9 GB 的图光复制就淹没了被测的编码时间 |
| 每个候选做**往返校验**（TIFF/PNG 逐位一致，JPEG 校验尺寸 + PSNR/SSIM 下限） | 「快但错」的参数组合没有意义 |
| 计时取多次的**最小值**，同时给出中位数 | GPU 上有偶发调度抖动，平均值会被一次抖动污染 |
| 先跑 `warmup` 再计时 | 首次调用包含 context/缓存建立，不代表稳态 |

## CLI 用法

```
icu_tune --input <file.bmp> --format tiff|png|jpeg [选项]

  --crop x,y,w,h           只取 BMP 的一块（大图强烈建议：读盘也省）
  --iters N                每个候选计时次数（默认 3，取最小值）
  --warmup N               预热次数（默认 1）
  --limit-mb N             显存上限（MB，默认不限制）
  --no-verify              跳过往返校验（不建议）
  --stop-on-error          某个候选失败就停止
  --json <path>            结果写成 JSON
  --markdown <path>        结果写成 Markdown 表格

扫描维度（不给则用默认候选集）：
  --tiles 0,128,256,512,1024   TIFF tile 边长（0 = 用 strip）
  --rows 8,32,64,128           TIFF strip 行数（tiles=0 时生效）
  --lzw-threads 128,256,512    TIFF LZW 每 block 线程数
  --batches 0,64,256           TIFF 每批上送单元数（0 = 按显存预算自动）
  --filters -1,0,1,2,3,4       PNG 行滤波（-1 = 自适应）
  --levels 1,6,9               PNG zlib 等级
  --png-threads 128,256,512    PNG 滤波 kernel 线程数
  --qualities 90,95            JPEG 质量
  --subsampling 444,420,422    JPEG 子采样
```

`--crop` 是**按需读取**：不会把整幅 1.9 GB 的 BMP 读进内存再裁剪。

## 实测样例（RTX 4070 / 11955 MiB / CUDA 13.2）

```
icu_tune --input data/Merge.bmp --crop 0,0,4096,3072 --format tiff \
         --tiles 256,512 --lzw-threads 128,256 --iters 1 --warmup 0
```

| # | 参数 | 耗时 (ms) | 吞吐 (MB/s) | 输出 (KB) | 显存峰值 (MB) |
| --- | --- | --- | --- | --- | --- |
| 1 | tile=256 lzw_threads=256 | 516 | 73.1 | 8712 | 93.0 |
| 2 | tile=256 lzw_threads=128 | 531 | 71.0 | 8712 | 93.0 |
| 3 | tile=512 lzw_threads=256 | 1205 | 31.3 | 8633 | 90.8 |
| 4 | tile=512 lzw_threads=128 | 1216 | 31.1 | 8633 | 90.8 |

结论：**tile 512 在这张图上比 256 慢一倍多**（LZW 是串行字典状态机，块越大每线程的工作
越串行、并行度越低），而压缩率只高 0.9%。这类结论必须在本机跑出来，不能靠猜。

显存上限同样可以在这里验证：

```
icu_tune --input data/Merge.bmp --crop 0,0,8192,8192 --tiles 512 --limit-mb 8
→ TIFF: CUDA LZW encode failed: 显存预算不足: 需要 25952256 字节, 上限 8388608 字节, 已用 0 字节
```

这是**预期行为**：小显存机器上要的是明确失败 + 可读原因，而不是把整机拖进 swap 或触发驱动重置。

## 把结论固化下来

调参结果是机器相关的，不要写进代码默认值里的「普适最优」。两种用法：

1. 一次性调优：用 `--json` 存档，按机器/显卡型号分目录保存；
2. 产品内调优：把结果填进 `image_codec_cuda_encode_params` 的
   `tiff_lzw_block_threads` / `tiff_lzw_unit_batch` / `png_block_threads` 字段（0 = 用库内默认）。
