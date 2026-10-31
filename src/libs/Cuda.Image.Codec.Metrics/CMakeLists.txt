# File: src/libs/Cuda.Image.Codec.Metrics/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

# 画质指标：MSE / PSNR / SSIM / MS-SSIM（GPU 并行，CPU 参考实现同库提供）。

# 定义与 ImageCodecOffload 其它后端（OpenCL / Vulkan）完全一致：
# 11x11 高斯窗(σ=1.5)、复制边缘、K1=0.01/K2=0.03/L=255、MS-SSIM 5 级权重
# 0.0448/0.2856/0.3001/0.2363/0.1333、2x2 均值下采样。三个后端必须给出相同的
# 数字，否则「换了后端画质变了」这种结论根本无从判断。
add_library(Cuda.Image.Codec.Metrics STATIC
  src/metrics.h
  src/metrics_common.h
  src/metrics_common.cpp
  src/metrics.cpp
  src/metrics_cpu.cpp
  src/metrics_launch.h
  src/metrics_launch.cu
  src/kernels/metrics.cu)

target_include_directories(Cuda.Image.Codec.Metrics PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(Cuda.Image.Codec.Metrics PUBLIC
  ImageCodecCUDA::Core
  ImageCodecCUDA::Common
  CUDA::cudart)

add_library(ImageCodecCUDA::Metrics ALIAS Cuda.Image.Codec.Metrics)
