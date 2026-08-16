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

struct GpuFusedTopKResult
{
    std::vector<Neighbor> neighbors;

    // Includes:
    //   L2 distance computation
    //   block-local Top-K selection
    float fusedKernelTimeMs = 0.0F;

    // Additional hierarchical candidate merge stages.
    float mergeKernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;

    // First fused local-selection stage + merge stages.
    int topKPasses = 0;
};

class GpuVectorIndex
{
public:
    static constexpr int MAX_TOP_K = 32;

    static constexpr int
        FUSED_CANDIDATES_PER_BLOCK = 256;

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

    // -----------------------------------------------------
    // Full-distance reference path
    // -----------------------------------------------------

    [[nodiscard]]
    GpuSearchResult search(
        const std::vector<float>& query,
        const GpuSearchConfig& config
    );

    // -----------------------------------------------------
    // Milestone 5 baseline
    //
    // Distance kernel
    //      ->
    // materialized N-distance array
    //      ->
    // hierarchical Top-K
    // -----------------------------------------------------

    [[nodiscard]]
    GpuTopKResult searchTopK(
        const std::vector<float>& query,
        int topK,
        const GpuSearchConfig& config
    );

    // -----------------------------------------------------
    // Milestone 6 fused path
    //
    // Fused distance + block-local Top-K
    //      ->
    // hierarchical candidate merge
    //
    // Does NOT materialize N distances on the production
    // fused search path.
    // -----------------------------------------------------

    [[nodiscard]]
    GpuFusedTopKResult searchTopKFused(
        const std::vector<float>& query,
        int topK
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

    void warmUpFusedTopK(
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

    // Retained for:
    //   - full-distance reference search
    //   - Milestone 5 baseline
    //
    // Milestone 6 fused search does not use this buffer.
    DeviceBuffer<float> deviceDistances_;

    DeviceBuffer<Neighbor> deviceTopKBufferA_;
    DeviceBuffer<Neighbor> deviceTopKBufferB_;

    cudaEvent_t distanceStart_{};
    cudaEvent_t distanceStop_{};

    cudaEvent_t topKStart_{};
    cudaEvent_t topKStop_{};

    double indexUploadTimeMs_ = 0.0;
};