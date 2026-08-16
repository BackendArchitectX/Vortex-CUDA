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

struct GpuSearchConfig
{
    DistanceKernel kernel =
        DistanceKernel::WarpShuffle;

    int blockSize = 256;
};

struct Neighbor
{
    int index = -1;
    float distance = 0.0F;
};

struct GpuSearchResult
{
    std::vector<float> distances;

    float kernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;
};

struct GpuTopKResult
{
    std::vector<Neighbor> neighbors;

    float distanceKernelTimeMs = 0.0F;
    float topKKernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;

    int topKPasses = 0;
};

class GpuVectorIndex
{
public:
    static constexpr int MAX_TOP_K = 32;

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
        const GpuSearchConfig& config
    );

    [[nodiscard]]
    GpuTopKResult searchTopK(
        const std::vector<float>& query,
        int topK,
        const GpuSearchConfig& config
    );

    void warmUp(
        const GpuSearchConfig& config,
        int iterations = 1
    );

    void warmUpTopK(
        const GpuSearchConfig& config,
        int topK,
        int iterations = 1
    );

    [[nodiscard]]
    static bool isSupportedBlockSize(
        int blockSize
    ) noexcept;

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

    DeviceBuffer<Neighbor> deviceTopKBufferA_;
    DeviceBuffer<Neighbor> deviceTopKBufferB_;

    cudaEvent_t distanceStart_{};
    cudaEvent_t distanceStop_{};

    cudaEvent_t topKStart_{};
    cudaEvent_t topKStop_{};

    double indexUploadTimeMs_ = 0.0;
};