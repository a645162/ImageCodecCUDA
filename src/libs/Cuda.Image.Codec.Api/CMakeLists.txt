# File: src/libs/Cuda.Image.Codec.Api/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

# 对外稳定 C API（include/ImageCodecCUDA/image_codec_cuda.h）实现库。
add_library(Cuda.Image.Codec.Api STATIC
  src/image_codec_cuda.cpp
  "${PROJECT_SOURCE_DIR}/include/ImageCodecCUDA/image_codec_cuda.h"
  "${PROJECT_SOURCE_DIR}/include/ImageCodecCUDA/image_codec_cuda.hpp")

target_include_directories(Cuda.Image.Codec.Api PUBLIC
  "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(Cuda.Image.Codec.Api PUBLIC
  ImageCodecCUDA::Core
  ImageCodecCUDA::Common
  ImageCodecCUDA::Tiff
  ImageCodecCUDA::NvJpeg
  ImageCodecCUDA::Png
  ImageCodecCUDA::Metrics)

# 共享库场景下 IMAGE_CODEC_CUDA_API 走 dllexport 分支；静态库下为空。
target_compile_definitions(Cuda.Image.Codec.Api PRIVATE
  IMAGE_CODEC_CUDA_BUILDING_LIB)

# 命名空间别名：target_link_libraries(app PRIVATE ImageCodecCUDA::Api)
add_library(ImageCodecCUDA::Api ALIAS Cuda.Image.Codec.Api)
