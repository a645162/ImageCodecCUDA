# src/ 下为多个独立静态库（参照 ImageCodecOpenCL 组织），用户可按需链接：
#   target_link_libraries(app PRIVATE ImageCodecCUDA::Tiff)   # 仅 TIFF（PUBLIC 传播 Core/Common）
add_subdirectory(libs)

# 全量聚合便捷入口（可选）：一次链接全部库（含 C API）。
add_library(ImageCodecCUDA INTERFACE)
target_include_directories(ImageCodecCUDA INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>")
target_link_libraries(ImageCodecCUDA INTERFACE
  ImageCodecCUDA::Core
  ImageCodecCUDA::Tiff
  ImageCodecCUDA::NvJpeg
  ImageCodecCUDA::Png
  ImageCodecCUDA::Api
  ImageCodecCUDA::Metrics
  ImageCodecCUDA::Tune)
add_library(ImageCodecCUDA::ImageCodecCUDA ALIAS ImageCodecCUDA)
