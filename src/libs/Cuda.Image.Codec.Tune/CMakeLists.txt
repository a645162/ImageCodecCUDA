# File: src/libs/Cuda.Image.Codec.Tune/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

# 参数扫描库：在自己的硬件上找到最优编码参数（tile / 批大小 / block 线程 / 滤波策略）。
#
# 只依赖各 codec 模块的 C++ 入口（不经过 C API），因此测量的是编码本身，
# 不含 C API 为了 ABI 稳定而做的那一次 host 缓冲复制。
add_library(Cuda.Image.Codec.Tune STATIC
  src/tune.h
  src/tune.cpp)

target_include_directories(Cuda.Image.Codec.Tune PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Cuda.Image.Codec.Tune PUBLIC
  ImageCodecCUDA::Core
  ImageCodecCUDA::Common
  ImageCodecCUDA::Tiff
  ImageCodecCUDA::Png
  ImageCodecCUDA::NvJpeg
  ImageCodecCUDA::Metrics)

# 命名空间别名：target_link_libraries(app PRIVATE ImageCodecCUDA::Tune)
add_library(ImageCodecCUDA::Tune ALIAS Cuda.Image.Codec.Tune)
