#pragma once

#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include <array>
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

    // 1 for the Milestone 7 path.
    // 1/2/4 for the adaptive Milestone 8/M10/M11 paths.
    int queryTileSize = 1;

    // True when a float4 path was used.
    // False for scalar fallback when dimension is not
    // divisible by four.
    bool vectorizedLoadsUsed = false;

    // True only for the Milestone 12 instruction-tuned
    // float4 kernel that uses explicit FP32 FMA accumulation
    // and shorter temporary live ranges.
    bool fmaTunedKernelUsed = false;

    float fusedKernelTimeMs = 0.0F;
    float mergeKernelTimeMs = 0.0F;

    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
    double totalTimeMs = 0.0;

    int topKPasses = 0;
};



struct GpuPipelinedTopKResult
{
    // Batch-major, then query-major, then rank-major:
    //
    // batch b / query q / rank r:
    //   neighbors[
    //       ((b * queryCount + q) * topK) + r
    //   ]
    std::vector<Neighbor> neighbors;

    int batchCount = 0;
    int queryCount = 0;
    int topK = 0;

    int streamCount = 0;
    int queryTileSize = 1;

    bool pinnedHostMemoryUsed = false;
    bool vectorizedLoadsUsed = false;
    bool fmaTunedKernelUsed = false;

    double totalTimeMs = 0.0;
    double averageBatchTimeMs = 0.0;
    double amortizedQueryTimeMs = 0.0;
    double throughputQueriesPerSecond = 0.0;
};

struct GpuQueryTileTuningResult
{
    int selectedTileSize = 1;
    int iterations = 0;

    // Median first-stage kernel latency for each candidate.
    // A negative value means the candidate was not eligible.
    float tile1MedianKernelTimeMs = -1.0F;
    float tile2MedianKernelTimeMs = -1.0F;
    float tile4MedianKernelTimeMs = -1.0F;
};

class GpuVectorIndex
{
public:
    static constexpr int MAX_TOP_K = 32;

    static constexpr int MAX_BATCH_SIZE = 32;

    static constexpr int MAX_QUERY_TILE_SIZE = 4;

    static constexpr int PIPELINE_STREAM_COUNT = 2;

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
    // One query is handled by each block.y slice.
    // -----------------------------------------------------

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatch(
        const std::vector<float>& queries,
        int queryCount,
        int topK
    );

    // -----------------------------------------------------
    // Milestone 8 query-tiled batched path
    //
    // One block reuses each loaded database value across
    // multiple queries.
    //
    // Adaptive query tiles:
    //   batch 1   -> tile 1
    //   batch 2-3 -> tile 2
    //   batch 4+  -> tile 4
    // -----------------------------------------------------

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiled(
        const std::vector<float>& queries,
        int queryCount,
        int topK
    );

    // -----------------------------------------------------
    // Milestone 10 vectorized query-tiled path
    //
    // Uses aligned float4 database/query loads when
    // dimension % 4 == 0. Otherwise it falls back to the
    // validated Milestone 8 scalar query-tiled kernel.
    // -----------------------------------------------------

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiledVectorized(
        const std::vector<float>& queries,
        int queryCount,
        int topK
    );


    // -----------------------------------------------------
    // Milestone 11 vectorized query-tile autotuning
    //
    // Candidate tiles are 1, 2 and 4, constrained by the
    // current batch size. Calibration is performed outside
    // steady-state timing and cached by (queryCount, topK).
    // -----------------------------------------------------

    [[nodiscard]]
    GpuQueryTileTuningResult autotuneVectorizedQueryTileSize(
        const std::vector<float>& queries,
        int queryCount,
        int topK,
        int iterations = 7
    );

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiledVectorizedAuto(
        const std::vector<float>& queries,
        int queryCount,
        int topK
    );

    [[nodiscard]]
    int tunedVectorizedQueryTileSize(
        int queryCount,
        int topK
    ) const noexcept;


    // -----------------------------------------------------
    // Milestone 12 instruction-tuned vectorized path
    //
    // Reuses the M11 hardware-selected query tile, while
    // replacing the float4 distance scan with an explicitly
    // FMA-based accumulator and shorter live ranges.
    // -----------------------------------------------------

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiledVectorizedFmaAuto(
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

    void warmUpTopKBatchTiled(
        int queryCount,
        int topK,
        int iterations = 1
    );

    void warmUpTopKBatchTiledVectorized(
        int queryCount,
        int topK,
        int iterations = 1
    );

    void warmUpTopKBatchTiledVectorizedAuto(
        int queryCount,
        int topK,
        int iterations = 1
    );

    void warmUpTopKBatchTiledVectorizedFmaAuto(
        int queryCount,
        int topK,
        int iterations = 1
    );


    // -----------------------------------------------------
    // Milestone 13 pinned-memory + dual-stream pipeline
    //
    // The input contains batchCount consecutive query batches.
    // Each query batch contains queryCount vectors.
    //
    // Two independent pipeline slots own:
    //   * pinned host input/output staging buffers,
    //   * device query/candidate buffers,
    //   * one non-blocking CUDA stream,
    //   * one completion event.
    //
    // H2D -> M12 kernel -> Top-K merge -> D2H remains ordered
    // inside each stream, while independent batches can overlap
    // across the two streams when the GPU supports it.
    // -----------------------------------------------------

    [[nodiscard]]
    GpuPipelinedTopKResult
    searchTopKBatchTiledVectorizedFmaPipelined(
        const std::vector<float>& batchedQueries,
        int batchCount,
        int queryCount,
        int topK
    );

    void warmUpTopKBatchTiledVectorizedFmaPipelined(
        int batchCount,
        int queryCount,
        int topK,
        int iterations = 1
    );

    [[nodiscard]]
    int asyncEngineCount() const noexcept
    {
        return asyncEngineCount_;
    }

    [[nodiscard]]
    bool concurrentKernelsSupported() const noexcept
    {
        return concurrentKernelsSupported_;
    }

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


    struct PipelineSlot;


    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiledVectorizedFmaWithTile(
        const std::vector<float>& queries,
        int queryCount,
        int topK,
        int queryTileSize
    );

    [[nodiscard]]
    GpuBatchTopKResult searchTopKBatchTiledVectorizedWithTile(
        const std::vector<float>& queries,
        int queryCount,
        int topK,
        int queryTileSize
    );

    [[nodiscard]]
    static constexpr std::size_t tuningCacheIndex(
        const int queryCount,
        const int topK) noexcept
    {
        return
            static_cast<std::size_t>(
                queryCount
            ) *
            static_cast<std::size_t>(
                MAX_TOP_K + 1
            ) +
            static_cast<std::size_t>(
                topK
            );
    }

    int vectorCount_;
    int dimension_;

    DeviceBuffer<float> deviceDatabase_;

    // Single-query storage.
    DeviceBuffer<float> deviceQuery_;
    DeviceBuffer<float> deviceDistances_;

    DeviceBuffer<Neighbor> deviceTopKBufferA_;
    DeviceBuffer<Neighbor> deviceTopKBufferB_;

    // Persistent batch storage shared by the M7, M8, M10,
    // M11 and M12 paths. The methods execute synchronously, so the same
    // ping-pong buffers can be reused safely.
    DeviceBuffer<float> deviceBatchQueries_;

    DeviceBuffer<Neighbor>
        deviceBatchTopKBufferA_;

    DeviceBuffer<Neighbor>
        deviceBatchTopKBufferB_;


    // 0 means "not tuned yet". Valid cached tiles are 1/2/4.
    std::array<
        int,
        static_cast<std::size_t>(
            MAX_BATCH_SIZE + 1
        ) *
        static_cast<std::size_t>(
            MAX_TOP_K + 1
        )
    > tunedVectorizedTileSizes_{};


    std::array<
        PipelineSlot*,
        PIPELINE_STREAM_COUNT
    > pipelineSlots_{};

    int asyncEngineCount_ = 0;
    bool concurrentKernelsSupported_ = false;

    cudaEvent_t distanceStart_{};
    cudaEvent_t distanceStop_{};

    cudaEvent_t topKStart_{};
    cudaEvent_t topKStop_{};

    double indexUploadTimeMs_ = 0.0;
};
