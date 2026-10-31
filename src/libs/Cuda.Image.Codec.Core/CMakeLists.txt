# CUDA 基础设施：设备/流/缓冲 RAII、错误文本、显存查询与预算记账、宿主侧计时。
add_library(Cuda.Image.Codec.Core STATIC
  src/cuda_error.h
  src/cuda_error.cpp
  src/cuda_buffer.h
  src/cuda_memory.h
  src/cuda_memory.cpp
  src/cuda_device.h
  src/cuda_device.cpp
  src/platform.h
  src/platform.cpp)

target_include_directories(Cuda.Image.Codec.Core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Cuda.Image.Codec.Core PUBLIC CUDA::cudart)

# 命名空间别名：target_link_libraries(app PRIVATE ImageCodecCUDA::Core)
add_library(ImageCodecCUDA::Core ALIAS Cuda.Image.Codec.Core)
