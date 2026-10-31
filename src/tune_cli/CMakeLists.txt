add_executable(icu_tune icu_tune.cpp)
target_link_libraries(icu_tune PRIVATE
  ImageCodecCUDA::Api
  ImageCodecCUDA::Tune
  ImageCodecCUDA::Metrics)
