cmake_minimum_required(VERSION 3.28)

# The P100 product targets Pascal sm_60 (Tesla P100, GP100) and is the primary architecture of
# this repository. The inherited Volta sm_70, Ampere sm_86 and early Ada sm_89 ports remain
# selectable during the migration. Set and validate the architecture before CUDA compiler
# detection so CMake never injects a generic host-toolchain default or accepts an unsupported
# target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(60|70|86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=60, 70, 86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# NINFER_PASCAL_BUILD gates every sm_60 route selection. GP100 has no tensor cores (no
# mma.sync/WMMA), no __dp4a (a 6.1 feature), but full-rate half2 HFMA2, 48 KiB shared memory
# per block, and 4 MiB L2. CUDA 13 removed Pascal compilation, so the build is pinned to the
# 12.x series; 12.4 is the floor because that is the cuBLAS behavior the P100 kernel routes are
# qualified against.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "60")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "Pascal sm_60 requires CUDA 12.x; CUDA 13 removed Pascal compilation")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    message(FATAL_ERROR "Pascal sm_60 requires CUDA 12.4 or newer; found "
                        "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  add_compile_definitions(NINFER_PASCAL_BUILD=1)
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "Volta sm_70 requires CUDA 12.8; CUDA 13 removed Volta compilation")
  endif()
  add_compile_definitions(NINFER_VOLTA_BUILD=1)
endif()

# The gguf/ggml_k translation units consume CUTLASS header-only types. sm_70 builds no CUTLASS
# TensorOp kernels here (the Volta ports carry their own); the fetch provides headers only.
# sm_60 needs no CUTLASS at all: gguf.cu gates its CUTLASS block out of Pascal builds and
# ggml_k.cu is replaced by Pascal stubs, so the offline-hostile fetch is skipped.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^70$")
  include(FetchContent)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples")
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cutlass)
endif()

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "60" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer's Volta/Ampere/Ada ports require CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  # CMake >= 3.27 ships no FindFFmpeg module, so the MSVC side locates an FFmpeg development
  # tree directly (include/ + lib/). Point NINFER_FFMPEG_ROOT (cache variable or environment
  # variable, or a CMAKE_PREFIX_PATH entry) at the extracted distribution
  # (e.g. gyan.dev ffmpeg-release-essentials or a vcpkg ffmpeg install).
  set(_ninfer_ffmpeg_hints
    "${NINFER_FFMPEG_ROOT}" $ENV{NINFER_FFMPEG_ROOT} ${CMAKE_PREFIX_PATH})
  find_path(NINFER_FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
    HINTS ${_ninfer_ffmpeg_hints} PATH_SUFFIXES include)
  set(NINFER_FFMPEG_LIBRARIES "")
  set(NINFER_FFMPEG_ALL_FOUND TRUE)
  foreach(_ninfer_ffmpeg_component avformat avcodec avutil swscale swresample)
    find_library(NINFER_FFMPEG_${_ninfer_ffmpeg_component}_LIBRARY
      NAMES ${_ninfer_ffmpeg_component}
      HINTS ${_ninfer_ffmpeg_hints} PATH_SUFFIXES lib)
    if(NOT NINFER_FFMPEG_${_ninfer_ffmpeg_component}_LIBRARY)
      set(NINFER_FFMPEG_ALL_FOUND FALSE)
    else()
      list(APPEND NINFER_FFMPEG_LIBRARIES
        "${NINFER_FFMPEG_${_ninfer_ffmpeg_component}_LIBRARY}")
    endif()
  endforeach()
  if(NINFER_FFMPEG_ALL_FOUND AND NINFER_FFMPEG_INCLUDE_DIR)
    add_library(ninfer_ffmpeg_dependencies INTERFACE)
    target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${NINFER_FFMPEG_INCLUDE_DIR})
    target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${NINFER_FFMPEG_LIBRARIES})
    set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
    message(STATUS "FFmpeg found at ${NINFER_FFMPEG_INCLUDE_DIR}")
  else()
    # Media decode needs FFmpeg headers/libs. Without them the text-only library slice
    # (core/artifact/ops/text) still configures and builds; media_decode, engine and apps
    # will not compile until FFmpeg is installed.
    message(WARNING "FFmpeg not found: ninfer_media_decode/ninfer_engine/apps cannot be built")
    add_library(ninfer_ffmpeg_stub INTERFACE)
    set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_stub)
  endif()
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 QUIET)
    if(CURL_FOUND)
      set(NINFER_CURL_TARGET CURL::libcurl)
      set(NINFER_HAVE_LIBCURL ON)
    else()
      message(WARNING "libcurl not found: media URL input will report RemoteUnavailable (Path/Data/Bytes sources still work)")
      add_library(ninfer_curl_stub INTERFACE)
      set(NINFER_CURL_TARGET ninfer_curl_stub)
    endif()
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
    set(NINFER_HAVE_LIBCURL ON)
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
