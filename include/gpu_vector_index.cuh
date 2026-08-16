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

    float fusedKernelTimeMs = 0.0F;
    float mergeKernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;

    int topKPasses = 0;
};

struct GpuBatchTopKResult
{
    // Query-major:
    //
    // query 0:
    //   neighbors[0 ... topK - 1]
    //
    // query 1:
    //   neighbors[topK ... 2 * topK - 1]
    std::vector<Neighbor> neighbors;

    int queryCount = 0;
    int topK = 0;

    float fusedKernelTimeMs = 0.0F;
    float mergeKernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;

    int topKPasses = 0;
};

class GpuVectorIndex
{
public:
    static constexpr int MAX_TOP_K = 32;

    static constexpr int MAX_BATCH_SIZE = 32;

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
    // -----------------------------------------------------

    [[nodiscard]]
    GpuTopKResult searchTopK(
        const std::vector<float>& query,
        int topK,
        const GpuSearchConfig& config
    );

    // -----------------------------------------------------
    // Milestone 6 optimized single-query path
    // -----------------------------------------------------

    [[nodiscard]]
    GpuFusedTopKResult searchTopKFused(
        const std::vector<float>& query,
        int topK
    );

    // -----------------------------------------------------
    // Milestone 7 batched fused path
    //
    // queries layout:
    //
    //   [Q0 dimensions...]
    //   [Q1 dimensions...]
    //   ...
    //
    // queries.size() must equal:
    //
    //   queryCount * dimension()
    // -----------------------------------------------------

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatch(
        const std::vector<float>& queries,
        int queryCount,
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

    void warmUpTopKBatch(
        int queryCount,
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

    // Single-query storage.
    DeviceBuffer<float> deviceQuery_;
    DeviceBuffer<float> deviceDistances_;

    DeviceBuffer<Neighbor> deviceTopKBufferA_;
    DeviceBuffer<Neighbor> deviceTopKBufferB_;

    // -----------------------------------------------------
    // Milestone 7 persistent batch storage.
    // -----------------------------------------------------

    DeviceBuffer<float> deviceBatchQueries_;

    DeviceBuffer<Neighbor>
        deviceBatchTopKBufferA_;

    DeviceBuffer<Neighbor>
        deviceBatchTopKBufferB_;

    cudaEvent_t distanceStart_{};
    cudaEvent_t distanceStop_{};

    cudaEvent_t topKStart_{};
    cudaEvent_t topKStop_{};

    double indexUploadTimeMs_ = 0.0;
};