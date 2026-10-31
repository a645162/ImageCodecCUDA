# File: src/tune_cli/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

add_executable(icu_tune icu_tune.cpp)
target_link_libraries(icu_tune PRIVATE
  ImageCodecCUDA::Api
  ImageCodecCUDA::Tune
  ImageCodecCUDA::Metrics)
