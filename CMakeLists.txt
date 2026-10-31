cmake_minimum_required(VERSION 3.24)

# nvcc 不一定在 PATH 上（例如只通过 /usr/local/cuda-13.2 安装）。这里在 project()
# 之前显式探测并写 CMAKE_CUDA_COMPILER：一旦 project() 因找不到编译器失败，
# 后续的 toolchain 探测都无从谈起（与主项目 ImageViewerQt 的做法一致）。
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(_icu_nvcc
    NAMES nvcc
    HINTS "${CUDAToolkit_ROOT}/bin" /usr/local/cuda/bin
    PATHS "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin" /usr/local/cuda-13.2/bin
    DOC "CUDA compiler")
  if(_icu_nvcc)
    set(CMAKE_CUDA_COMPILER "${_icu_nvcc}" CACHE FILEPATH "CUDA compiler")
    get_filename_component(_icu_cuda_root "${_icu_nvcc}" DIRECTORY)
    get_filename_component(_icu_cuda_root "${_icu_cuda_root}" DIRECTORY)
    if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "${_icu_cuda_root}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "${_icu_cuda_root}" CACHE PATH "CUDA Toolkit root")
    endif()
  endif()
endif()

project(ImageCodecCUDA LANGUAGES C CXX CUDA)

# ---- vcpkg toolchain bootstrap（与 ImageCodecOpenCL / ImageCodecVulkan 相同模式）----
# 独立构建时若未显式传入 CMAKE_TOOLCHAIN_FILE，则按 VCPKG_ROOT / C:/vcpkg 自动探测。
# 作为 add_subdirectory() 子项目复用时，宿主工程的 toolchain 已经生效，这里不会覆盖。
if(WIN32)
  set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg triplet" FORCE)
endif()
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain")
  elseif(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain")
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 源码为 UTF-8（含中文注释）；MSVC 默认按系统代码页解析会乱码并破坏词法。
if(MSVC)
  add_compile_options(/utf-8)
endif()

# ---- CUDA 架构 ----
# 默认值按 nvcc 版本决定，而不是写死一份列表：
#   * nvcc < 13 ：Pascal(10 系, sm_61) 仍可离线编译，默认一并带上；
#   * nvcc >= 13：Pascal/Maxwell/Volta 已被工具链移除，从 Turing(sm_75) 起，
#                 即 20 系 → 50 系 = 75;80;86;89;120；
#   * Volta(sm_70/72) 任何情况下都不默认包含（数据中心的 V100 不在此列）。
# 最后再与本机 nvcc 实际支持的架构取交集：CUDA 12.0~12.7 不认识 sm_120，
# 硬塞进去会直接 configure 失败。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(_icu_arch_wanted 75 80 86 89 120)
    set(_icu_arch_reason "nvcc ${CMAKE_CUDA_COMPILER_VERSION} >= 13：Pascal/Volta 已移除，自 Turing 起")
  else()
    set(_icu_arch_wanted 61 75 80 86 89 120)
    set(_icu_arch_reason "nvcc ${CMAKE_CUDA_COMPILER_VERSION} < 13：附带 Pascal(sm_61, 10 系)，不含 Volta")
  endif()

  # --list-gpu-arch 输出形如 "compute_75"。老版本 nvcc 不一定支持该参数，
  # 取不到就退化为「全部采用」，由 nvcc 自己报错而不是我们瞎猜。
  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
    OUTPUT_VARIABLE _icu_arch_list
    ERROR_QUIET
    RESULT_VARIABLE _icu_arch_rc)
  set(_icu_arch_supported "")
  if(_icu_arch_rc EQUAL 0 AND _icu_arch_list)
    string(REGEX MATCHALL "compute_[0-9]+" _icu_arch_tokens "${_icu_arch_list}")
    foreach(_t IN LISTS _icu_arch_tokens)
      string(REPLACE "compute_" "" _n "${_t}")
      list(APPEND _icu_arch_supported "${_n}")
    endforeach()
  endif()

  set(_icu_arch_final "")
  foreach(_a IN LISTS _icu_arch_wanted)
    if(_icu_arch_supported AND NOT _a IN_LIST _icu_arch_supported)
      continue()
    endif()
    list(APPEND _icu_arch_final "${_a}")
  endforeach()
  if(NOT _icu_arch_final)
    set(_icu_arch_final "native")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_icu_arch_final}" CACHE STRING "CUDA architectures" FORCE)
  message(STATUS "ImageCodecCUDA: CUDA 架构 = ${CMAKE_CUDA_ARCHITECTURES}（${_icu_arch_reason}）")
else()
  message(STATUS "ImageCodecCUDA: CUDA 架构 = ${CMAKE_CUDA_ARCHITECTURES}（外部指定）")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

# nvJPEG 属于 CUDA Toolkit 自带组件；缺失时 JPEG 后端以显式 UNSUPPORTED 返回，
# 而不是悄悄改用别的实现（上层需要知道实际走了哪条路径）。
option(ICUDA_ENABLE_NVJPEG "Enable nvJPEG JPEG codec" ON)
if(ICUDA_ENABLE_NVJPEG AND NOT TARGET CUDA::nvjpeg)
  message(WARNING "CUDA::nvjpeg 未找到，JPEG 编解码将报告 UNSUPPORTED（关闭 ICUDA_ENABLE_NVJPEG 可消除本告警）")
endif()

option(ICUDA_BUILD_CLI "Build test CLI" ON)
option(ICUDA_BUILD_TESTS "Build ctest suite" ON)
option(ICUDA_BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(src)
if(ICUDA_BUILD_CLI)
  add_subdirectory(src/cli)
  add_subdirectory(src/tune_cli)   # icu_tune：参数扫描
endif()
if(ICUDA_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(ICUDA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
