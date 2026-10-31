# Nvidia.Image.Codec.NvJpeg —— NVIDIA 固定功能硬件编解码（NVJPG）封装。
#
# 与 ImageCodecPlatform 的分工：Intel 的 oneVPL/QuickSync 是另一家厂商的硬件路径，
# 归 ImageCodecPlatform 的 Intel.Image.Codec.Jpeg.oneVPL；NVIDIA 的 nvJPEG 由本模块
# 承载，只依赖 CUDA Toolkit 的 nvjpeg 组件（不需要 nvcc 编译本模块）。
add_library(Nvidia.Image.Codec.NvJpeg STATIC
  src/jpeg_codec.h
  src/jpeg_codec.cpp)

target_include_directories(Nvidia.Image.Codec.NvJpeg PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Nvidia.Image.Codec.NvJpeg PUBLIC
  ImageCodecCUDA::Core
  ImageCodecCUDA::Common
  CUDA::cudart)

if(TARGET CUDA::nvjpeg AND ICUDA_ENABLE_NVJPEG)
  target_compile_definitions(Nvidia.Image.Codec.NvJpeg PRIVATE ICUDA_HAS_NVJPEG=1)
  target_link_libraries(Nvidia.Image.Codec.NvJpeg PUBLIC CUDA::nvjpeg)
endif()

# 命名空间别名：target_link_libraries(app PRIVATE ImageCodecCUDA::NvJpeg)
add_library(ImageCodecCUDA::NvJpeg ALIAS Nvidia.Image.Codec.NvJpeg)
