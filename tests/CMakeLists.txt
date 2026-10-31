# File: tests/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

add_executable(codec_roundtrip_test codec_roundtrip_test.cpp)
target_link_libraries(codec_roundtrip_test PRIVATE
  ImageCodecCUDA::Api
  ImageCodecCUDA::Tiff
  ImageCodecCUDA::Png
  ImageCodecCUDA::NvJpeg
  ImageCodecCUDA::Common)
add_test(NAME codec_roundtrip_test COMMAND codec_roundtrip_test)
set_tests_properties(codec_roundtrip_test PROPERTIES SKIP_RETURN_CODE 2)
