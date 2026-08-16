#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

class CudaException final : public std::runtime_error
{
public:
    CudaException(
        cudaError_t error,
        const char* expression,
        const char* file,
        int line)
        : std::runtime_error(
            buildMessage(
                error,
                expression,
                file,
                line
            )
        ),
          error_(error)
    {
    }

    [[nodiscard]]
    cudaError_t error() const noexcept
    {
        return error_;
    }

private:
    static std::string buildMessage(
        cudaError_t error,
        const char* expression,
        const char* file,
        int line)
    {
        std::ostringstream stream;

        stream
            << "CUDA call failed ["
            << cudaGetErrorName(error)
            << "]: "
            << cudaGetErrorString(error)
            << "; expression="
            << expression
            << "; file="
            << file
            << "; line="
            << line;

        return stream.str();
    }

    cudaError_t error_;
};

inline void vortexCudaCheck(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line)
{
    if (error != cudaSuccess)
    {
        throw CudaException(
            error,
            expression,
            file,
            line
        );
    }
}

#define CUDA_CHECK(call)                                                   \
    ::vortexCudaCheck(                                                     \
        (call),                                                            \
        #call,                                                             \
        __FILE__,                                                          \
        __LINE__                                                           \
    )
