cmake_minimum_required(VERSION 3.24)

project(
        VortexCUDA
        VERSION 0.2.0
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3050 Laptop GPU
# Compute Capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(
        vortex_cuda
        src/main.cu
        src/gpu_vector_index.cu
)

target_include_directories(
        vortex_cuda
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(
        vortex_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

if (MSVC)
    target_compile_options(
            vortex_cuda
            PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
endif()