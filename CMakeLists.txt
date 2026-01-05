cmake_minimum_required(VERSION 4.2)
project(SpriteExtractor C)

set(CMAKE_C_STANDARD 23)

add_executable(SpriteExtractor main.c)

##

set(MI_SECURE ON)
set(MI_OVERRIDE ON)
set(MI_BUILD_SHARED OFF)
set(MI_OPT_ARCH ON)
set(MI_OPT_SIMD ON)
set(MI_BUILD_TESTS OFF)
set(MI_BUILD_BENCHMARKS OFF)

##

target_include_directories(SpriteExtractor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/stb")

include(FetchContent)
FetchContent_Declare(
        mimalloc
        GIT_REPOSITORY
        https://github.com/microsoft/mimalloc.git
        GIT_TAG
        v3.0.11)

FetchContent_MakeAvailable(mimalloc)

##

target_link_libraries(SpriteExtractor PRIVATE mimalloc-static)

target_compile_options(SpriteExtractor PRIVATE -march=native -Os -fopenmp -mfpmath=both -ffast-math)
target_link_options(SpriteExtractor PRIVATE -flto -fstack-clash-protection -fPIE -W -fopenmp)
