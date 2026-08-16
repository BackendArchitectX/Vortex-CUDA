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

# ---------------------------------------------------------
# Build metadata captured when CMake configures the project.
# This is included in M15 benchmark reports.
# ---------------------------------------------------------

set(VORTEX_GIT_COMMIT "unknown")
set(VORTEX_GIT_DIRTY 1)

find_package(Git QUIET)

if (GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VORTEX_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VORTEX_GIT_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if (VORTEX_GIT_STATUS STREQUAL "")
        set(VORTEX_GIT_DIRTY 0)
    else()
        set(VORTEX_GIT_DIRTY 1)
    endif()
endif()

# ---------------------------------------------------------
# Main Vortex executable
# ---------------------------------------------------------

add_executable(
        vortex_cuda
        src/main.cu
        src/gpu_vector_index.cu
)

# ---------------------------------------------------------
# Milestone 15 production benchmark executable
# ---------------------------------------------------------

add_executable(
        vortex_benchmark
        src/benchmark_m15.cu
        src/gpu_vector_index.cu
)

foreach(VORTEX_TARGET IN ITEMS vortex_cuda vortex_benchmark)
    target_include_directories(
            ${VORTEX_TARGET}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(
            ${VORTEX_TARGET}
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    if (MSVC)
        target_compile_options(
                ${VORTEX_TARGET}
                PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
        )
    endif()
endforeach()

target_compile_definitions(
        vortex_benchmark
        PRIVATE
        VORTEX_GIT_COMMIT="${VORTEX_GIT_COMMIT}"
        VORTEX_GIT_DIRTY=${VORTEX_GIT_DIRTY}
)
