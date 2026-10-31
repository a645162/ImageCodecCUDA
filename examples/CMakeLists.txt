add_executable(icu_c_example c_example.c)
target_link_libraries(icu_c_example PRIVATE ImageCodecCUDA::Api)

add_executable(icu_cpp_example cpp_example.cpp)
target_link_libraries(icu_cpp_example PRIVATE ImageCodecCUDA::Api)
