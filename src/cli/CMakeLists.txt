# File: src/cli/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

add_executable(icu main.cpp)
target_link_libraries(icu PRIVATE ImageCodecCUDA::Api)
set_target_properties(icu PROPERTIES OUTPUT_NAME icu)
