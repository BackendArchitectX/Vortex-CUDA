#pragma once

#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

enum class DistanceKernel
{
    SharedMemory,
    WarpShuffle
};

struct GpuSearchResult
{
    std::vector<float> distances;

    float kernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;
};

class GpuVectorIndex
{
public:
    GpuVectorIndex(
        const std::vector<float>& database,
        int vectorCount,
        int dimension
    );

    ~GpuVectorIndex();

    GpuVectorIndex(
        const GpuVectorIndex&
    ) = delete;

    GpuVectorIndex& operator=(
        const GpuVectorIndex&
    ) = delete;

    GpuVectorIndex(
        GpuVectorIndex&&
    ) = delete;

    GpuVectorIndex& operator=(
        GpuVectorIndex&&
    ) = delete;

    [[nodiscard]]
    GpuSearchResult search(
        const std::vector<float>& query,
        DistanceKernel kernel
    );

    void warmUp(
        DistanceKernel kernel,
        int iterations = 1
    );

    [[nodiscard]]
    int vectorCount() const noexcept
    {
        return vectorCount_;
    }

    [[nodiscard]]
    int dimension() const noexcept
    {
        return dimension_;
    }

    [[nodiscard]]
    double indexUploadTimeMs() const noexcept
    {
        return indexUploadTimeMs_;
    }

    [[nodiscard]]
    std::size_t indexBytes() const noexcept
    {
        return
            static_cast<std::size_t>(
                vectorCount_
            ) *
            static_cast<std::size_t>(
                dimension_
            ) *
            sizeof(float);
    }

private:
    int vectorCount_;
    int dimension_;

    DeviceBuffer<float> deviceDatabase_;
    DeviceBuffer<float> deviceQuery_;
    DeviceBuffer<float> deviceDistances_;

    cudaEvent_t kernelStart_{};
    cudaEvent_t kernelStop_{};

    double indexUploadTimeMs_ = 0.0;
};