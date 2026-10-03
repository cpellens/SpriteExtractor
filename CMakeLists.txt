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
    v3.5.4)

FetchContent_MakeAvailable(mimalloc)

##

target_link_libraries(SpriteExtractor PRIVATE mimalloc-static)

find_package(OpenMP)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(SpriteExtractor PRIVATE
        -O3
        -DNDEBUG
        -flto=thin
        -march=native
        -ffast-math
        -fno-omit-frame-pointer
        -Rpass=loop-vectorize
        -Rpass-missed=loop-vectorize
        -Rpass-analysis=loop-vectorize
        -Wpadded
    )
    if(WIN32)
        target_link_options(SpriteExtractor PRIVATE -flto=thin -fuse-ld=lld)
    else()
        target_link_options(SpriteExtractor PRIVATE -flto=thin)
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    target_compile_options(SpriteExtractor PRIVATE
        -O3
        -DNDEBUG
        -flto=auto
        -march=native
        -ffast-math
        -fno-omit-frame-pointer
        -fopt-info-vec-all
        -Wpadded
    )
    target_link_options(SpriteExtractor PRIVATE -flto=auto)
else()
    target_compile_options(SpriteExtractor PRIVATE -O3 -DNDEBUG -march=native -ffast-math)
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(SpriteExtractor PRIVATE OpenMP::OpenMP_C)
endif()
