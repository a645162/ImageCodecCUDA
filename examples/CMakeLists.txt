# File: examples/CMakeLists.txt
# Brief: CMake build configuration.
# Author: Haomin Kong
# Email: a645162@163.com
# Created: 2026-10-31

add_executable(icu_c_example c_example.c)
target_link_libraries(icu_c_example PRIVATE ImageCodecCUDA::Api)

add_executable(icu_cpp_example cpp_example.cpp)
target_link_libraries(icu_cpp_example PRIVATE ImageCodecCUDA::Api)
