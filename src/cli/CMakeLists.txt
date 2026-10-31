add_executable(icu main.cpp)
target_link_libraries(icu PRIVATE ImageCodecCUDA::Api)
set_target_properties(icu PROPERTIES OUTPUT_NAME icu)
