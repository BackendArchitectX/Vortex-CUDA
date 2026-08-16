#include "gpu_vector_index.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int CUDA_WARP_SIZE = 32;

constexpr int TOPK_BLOCK_SIZE = 256;

constexpr int FUSED_GROUP_SIZE =
    GpuVectorIndex::FUSED_CANDIDATES_PER_BLOCK;

static_assert(
    TOPK_BLOCK_SIZE ==
    FUSED_GROUP_SIZE
);

// =========================================================
// Utility
// =========================================================

std::size_t requirePositive(
    const int value,
    const char* name)
{
    if (value <= 0)
    {
        throw std::invalid_argument(
            std::string(name) +
            " must be positive."
        );
    }

    return
        static_cast<std::size_t>(
            value
        );
}

std::size_t ceilDivide(
    const std::size_t value,
    const std::size_t divisor)
{
    return
        (value + divisor - 1) /
        divisor;
}

std::size_t validateDatabase(
    const std::vector<float>& database,
    const int vectorCount,
    const int dimension)
{
    const std::size_t vectors =
        requirePositive(
            vectorCount,
            "vectorCount"
        );

    const std::size_t dimensions =
        requirePositive(
            dimension,
            "dimension"
        );

    if (vectors >
        std::numeric_limits<
            std::size_t
        >::max() /
        dimensions)
    {
        throw std::overflow_error(
            "Database dimensions overflow size_t."
        );
    }

    const std::size_t expectedElements =
        vectors *
        dimensions;

    if (database.size() !=
        expectedElements)
    {
        throw std::invalid_argument(
            "Database size does not match "
            "vectorCount * dimension."
        );
    }

    return expectedElements;
}

std::size_t calculateTopKBufferCapacity(
    const int vectorCount)
{
    const std::size_t vectors =
        requirePositive(
            vectorCount,
            "vectorCount"
        );

    const std::size_t blockCount =
        ceilDivide(
            vectors,
            static_cast<std::size_t>(
                TOPK_BLOCK_SIZE
            )
        );

    if (blockCount >
        std::numeric_limits<
            std::size_t
        >::max() /
        static_cast<std::size_t>(
            GpuVectorIndex::MAX_TOP_K
        ))
    {
        throw std::overflow_error(
            "Top-K intermediate buffer size overflow."
        );
    }

    return
        blockCount *
        static_cast<std::size_t>(
            GpuVectorIndex::MAX_TOP_K
        );
}

} // namespace

// =========================================================
// Warp reduction
// =========================================================

__inline__ __device__
float warpReduceSum(
    float value)
{
    constexpr unsigned int FULL_MASK =
        0xFFFFFFFFU;

    for (int offset =
             CUDA_WARP_SIZE / 2;
         offset > 0;
         offset >>= 1)
    {
        value +=
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );
    }

    return value;
}

// =========================================================
// Shared-memory distance kernel
// =========================================================

template <int BLOCK_SIZE>
__global__ void l2DistanceSharedKernel(
    const float* __restrict__ database,
    const float* __restrict__ query,
    float* __restrict__ distances,
    const int vectorCount,
    const int dimension)
{
    const int vectorIndex =
        static_cast<int>(
            blockIdx.x
        );

    if (vectorIndex >=
        vectorCount)
    {
        return;
    }

    extern __shared__ float partial[];

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const float* vector =
        database +
        static_cast<std::size_t>(
            vectorIndex
        ) *
        static_cast<std::size_t>(
            dimension
        );

    float localSum = 0.0F;

    for (int dimensionIndex =
             threadId;
         dimensionIndex <
             dimension;
         dimensionIndex +=
             BLOCK_SIZE)
    {
        const float difference =
            vector[
                dimensionIndex
            ] -
            query[
                dimensionIndex
            ];

        localSum +=
            difference *
            difference;
    }

    partial[
        threadId
    ] =
        localSum;

    __syncthreads();

    for (unsigned int stride =
             BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1)
    {
        if (threadIdx.x <
            stride)
        {
            partial[
                threadIdx.x
            ] +=
                partial[
                    threadIdx.x +
                    stride
                ];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        distances[
            vectorIndex
        ] =
            partial[0];
    }
}

// =========================================================
// Warp-shuffle distance kernel
// =========================================================

template <int BLOCK_SIZE>
__global__ void l2DistanceWarpKernel(
    const float* __restrict__ database,
    const float* __restrict__ query,
    float* __restrict__ distances,
    const int vectorCount,
    const int dimension)
{
    constexpr int WARPS_PER_BLOCK =
        BLOCK_SIZE /
        CUDA_WARP_SIZE;

    const int vectorIndex =
        static_cast<int>(
            blockIdx.x
        );

    if (vectorIndex >=
        vectorCount)
    {
        return;
    }

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const int laneId =
        threadId &
        (CUDA_WARP_SIZE - 1);

    const int warpId =
        threadId /
        CUDA_WARP_SIZE;

    __shared__ float warpSums[
        WARPS_PER_BLOCK
    ];

    const float* vector =
        database +
        static_cast<std::size_t>(
            vectorIndex
        ) *
        static_cast<std::size_t>(
            dimension
        );

    float localSum = 0.0F;

    for (int dimensionIndex =
             threadId;
         dimensionIndex <
             dimension;
         dimensionIndex +=
             BLOCK_SIZE)
    {
        const float difference =
            vector[
                dimensionIndex
            ] -
            query[
                dimensionIndex
            ];

        localSum +=
            difference *
            difference;
    }

    localSum =
        warpReduceSum(
            localSum
        );

    if (laneId == 0)
    {
        warpSums[
            warpId
        ] =
            localSum;
    }

    __syncthreads();

    if (warpId == 0)
    {
        float blockSum =
            laneId <
            WARPS_PER_BLOCK
                ? warpSums[
                    laneId
                ]
                : 0.0F;

        blockSum =
            warpReduceSum(
                blockSum
            );

        if (laneId == 0)
        {
            distances[
                vectorIndex
            ] =
                blockSum;
        }
    }
}

// =========================================================
// Top-K helpers
// =========================================================

__inline__ __device__
Neighbor makeInvalidNeighbor()
{
    Neighbor neighbor;

    // +infinity
    neighbor.distance =
        __int_as_float(
            0x7F800000
        );

    neighbor.index =
        INT_MAX;

    return neighbor;
}

__inline__ __device__
bool neighborLess(
    const Neighbor& left,
    const Neighbor& right)
{
    if (left.distance <
        right.distance)
    {
        return true;
    }

    if (left.distance >
        right.distance)
    {
        return false;
    }

    return
        left.index <
        right.index;
}

template <int BLOCK_SIZE>
__device__ void bitonicSortNeighbors(
    Neighbor* values)
{
    const unsigned int threadId =
        static_cast<unsigned int>(
            threadIdx.x
        );

    for (unsigned int sequenceSize = 2;
         sequenceSize <=
             BLOCK_SIZE;
         sequenceSize <<= 1)
    {
        for (unsigned int stride =
                 sequenceSize >> 1;
             stride > 0;
             stride >>= 1)
        {
            const unsigned int partner =
                threadId ^
                stride;

            if (partner >
                threadId)
            {
                const bool ascending =
                    (
                        threadId &
                        sequenceSize
                    ) == 0;

                const Neighbor left =
                    values[
                        threadId
                    ];

                const Neighbor right =
                    values[
                        partner
                    ];

                const bool shouldSwap =
                    ascending
                        ? neighborLess(
                            right,
                            left
                        )
                        : neighborLess(
                            left,
                            right
                        );

                if (shouldSwap)
                {
                    values[
                        threadId
                    ] =
                        right;

                    values[
                        partner
                    ] =
                        left;
                }
            }

            __syncthreads();
        }
    }
}

// =========================================================
// Milestone 5 stage 1
//
// Full distance array -> local Top-K candidates.
// =========================================================

__global__ void topKFromDistancesKernel(
    const float* __restrict__ distances,
    Neighbor* __restrict__ output,
    const int count,
    const int topK)
{
    __shared__ Neighbor candidates[
        TOPK_BLOCK_SIZE
    ];

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const std::size_t globalIndex =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        static_cast<std::size_t>(
            TOPK_BLOCK_SIZE
        ) +
        static_cast<std::size_t>(
            threadId
        );

    Neighbor candidate =
        makeInvalidNeighbor();

    if (globalIndex <
        static_cast<std::size_t>(
            count
        ))
    {
        candidate.index =
            static_cast<int>(
                globalIndex
            );

        candidate.distance =
            distances[
                globalIndex
            ];
    }

    candidates[
        threadId
    ] =
        candidate;

    __syncthreads();

    bitonicSortNeighbors<
        TOPK_BLOCK_SIZE
    >(
        candidates
    );

    if (threadId <
        topK)
    {
        const std::size_t outputIndex =
            static_cast<std::size_t>(
                blockIdx.x
            ) *
            static_cast<std::size_t>(
                topK
            ) +
            static_cast<std::size_t>(
                threadId
            );

        output[
            outputIndex
        ] =
            candidates[
                threadId
            ];
    }
}

// =========================================================
// Candidate merge stage
// =========================================================

__global__ void topKFromCandidatesKernel(
    const Neighbor* __restrict__ input,
    Neighbor* __restrict__ output,
    const int count,
    const int topK)
{
    __shared__ Neighbor candidates[
        TOPK_BLOCK_SIZE
    ];

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const std::size_t globalIndex =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        static_cast<std::size_t>(
            TOPK_BLOCK_SIZE
        ) +
        static_cast<std::size_t>(
            threadId
        );

    Neighbor candidate =
        makeInvalidNeighbor();

    if (globalIndex <
        static_cast<std::size_t>(
            count
        ))
    {
        candidate =
            input[
                globalIndex
            ];
    }

    candidates[
        threadId
    ] =
        candidate;

    __syncthreads();

    bitonicSortNeighbors<
        TOPK_BLOCK_SIZE
    >(
        candidates
    );

    if (threadId <
        topK)
    {
        const std::size_t outputIndex =
            static_cast<std::size_t>(
                blockIdx.x
            ) *
            static_cast<std::size_t>(
                topK
            ) +
            static_cast<std::size_t>(
                threadId
            );

        output[
            outputIndex
        ] =
            candidates[
                threadId
            ];
    }
}

// =========================================================
// Milestone 6 fused kernel
//
// One 256-thread block processes 256 vectors.
//
// There are 8 warps in the block.
//
// Each warp processes 32 vectors sequentially:
//
//   8 warps * 32 vectors = 256 vectors/block
//
// A warp cooperatively calculates the L2 distance for one
// vector, then lane 0 stores that vector's candidate into
// shared memory.
//
// After all 256 vectors are processed, the same block
// performs local Top-K selection.
//
// No N-sized global distance array is written.
// =========================================================

__global__ void fusedDistanceLocalTopKKernel(
    const float* __restrict__ database,
    const float* __restrict__ query,
    Neighbor* __restrict__ output,
    const int vectorCount,
    const int dimension,
    const int topK)
{
    constexpr int WARPS_PER_BLOCK =
        TOPK_BLOCK_SIZE /
        CUDA_WARP_SIZE;

    static_assert(
        WARPS_PER_BLOCK == 8
    );

    __shared__ Neighbor candidates[
        FUSED_GROUP_SIZE
    ];

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const int laneId =
        threadId &
        (CUDA_WARP_SIZE - 1);

    const int warpId =
        threadId /
        CUDA_WARP_SIZE;

    // -----------------------------------------------------
    // Initialize every shared-memory candidate first.
    //
    // The barrier is important because a different warp's
    // lane zero may later overwrite the same shared slot.
    // -----------------------------------------------------

    candidates[
        threadId
    ] =
        makeInvalidNeighbor();

    __syncthreads();

    const std::size_t blockVectorBase =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        static_cast<std::size_t>(
            FUSED_GROUP_SIZE
        );

    // -----------------------------------------------------
    // Each warp owns local candidate positions:
    //
    // warp 0 -> 0, 8, 16, ...
    // warp 1 -> 1, 9, 17, ...
    // ...
    // warp 7 -> 7, 15, 23, ...
    // -----------------------------------------------------

    for (int localVectorIndex =
             warpId;
         localVectorIndex <
             FUSED_GROUP_SIZE;
         localVectorIndex +=
             WARPS_PER_BLOCK)
    {
        const std::size_t vectorIndex =
            blockVectorBase +
            static_cast<std::size_t>(
                localVectorIndex
            );

        float localSum = 0.0F;

        if (vectorIndex <
            static_cast<std::size_t>(
                vectorCount
            ))
        {
            const float* vector =
                database +
                vectorIndex *
                static_cast<std::size_t>(
                    dimension
                );

            // ---------------------------------------------
            // One warp cooperatively evaluates one vector.
            //
            // For dimension 128:
            // each lane processes four values.
            // ---------------------------------------------

            for (int dimensionIndex =
                     laneId;
                 dimensionIndex <
                     dimension;
                 dimensionIndex +=
                     CUDA_WARP_SIZE)
            {
                const float difference =
                    vector[
                        dimensionIndex
                    ] -
                    query[
                        dimensionIndex
                    ];

                localSum +=
                    difference *
                    difference;
            }
        }

        // Every lane participates.
        localSum =
            warpReduceSum(
                localSum
            );

        if (laneId == 0 &&
            vectorIndex <
                static_cast<std::size_t>(
                    vectorCount
                ))
        {
            candidates[
                localVectorIndex
            ] =
                Neighbor{
                    static_cast<int>(
                        vectorIndex
                    ),
                    localSum
                };
        }
    }

    // Every candidate must be complete before sorting.
    __syncthreads();

    // -----------------------------------------------------
    // Local exact Top-K over 256 vector candidates.
    // -----------------------------------------------------

    bitonicSortNeighbors<
        TOPK_BLOCK_SIZE
    >(
        candidates
    );

    if (threadId <
        topK)
    {
        const std::size_t outputIndex =
            static_cast<std::size_t>(
                blockIdx.x
            ) *
            static_cast<std::size_t>(
                topK
            ) +
            static_cast<std::size_t>(
                threadId
            );

        output[
            outputIndex
        ] =
            candidates[
                threadId
            ];
    }
}

// =========================================================
// Host-side launch helpers
// =========================================================

namespace
{

template <int BLOCK_SIZE>
void launchSharedKernel(
    const float* database,
    const float* query,
    float* distances,
    const int vectorCount,
    const int dimension)
{
    constexpr std::size_t
        SHARED_MEMORY_BYTES =
            static_cast<std::size_t>(
                BLOCK_SIZE
            ) *
            sizeof(float);

    l2DistanceSharedKernel<
        BLOCK_SIZE
    ><<<
        vectorCount,
        BLOCK_SIZE,
        SHARED_MEMORY_BYTES
    >>>(
        database,
        query,
        distances,
        vectorCount,
        dimension
    );
}

template <int BLOCK_SIZE>
void launchWarpKernel(
    const float* database,
    const float* query,
    float* distances,
    const int vectorCount,
    const int dimension)
{
    l2DistanceWarpKernel<
        BLOCK_SIZE
    ><<<
        vectorCount,
        BLOCK_SIZE
    >>>(
        database,
        query,
        distances,
        vectorCount,
        dimension
    );
}

void launchConfiguredKernel(
    const GpuSearchConfig& config,
    const float* database,
    const float* query,
    float* distances,
    const int vectorCount,
    const int dimension)
{
    switch (config.kernel)
    {
        case DistanceKernel::SharedMemory:
        {
            switch (config.blockSize)
            {
                case 32:
                    launchSharedKernel<32>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 64:
                    launchSharedKernel<64>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 128:
                    launchSharedKernel<128>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 256:
                    launchSharedKernel<256>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                default:
                    throw std::invalid_argument(
                        "Unsupported shared-memory "
                        "block size."
                    );
            }

            break;
        }

        case DistanceKernel::WarpShuffle:
        {
            switch (config.blockSize)
            {
                case 32:
                    launchWarpKernel<32>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 64:
                    launchWarpKernel<64>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 128:
                    launchWarpKernel<128>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                case 256:
                    launchWarpKernel<256>(
                        database,
                        query,
                        distances,
                        vectorCount,
                        dimension
                    );
                    break;

                default:
                    throw std::invalid_argument(
                        "Unsupported warp-shuffle "
                        "block size."
                    );
            }

            break;
        }

        default:
            throw std::invalid_argument(
                "Unknown distance kernel."
            );
    }
}

struct CandidateMergeResult
{
    Neighbor* finalBuffer = nullptr;
    int mergePasses = 0;
};

CandidateMergeResult mergeTopKCandidates(
    Neighbor* currentBuffer,
    Neighbor* nextBuffer,
    std::size_t currentCount,
    const int topK)
{
    CandidateMergeResult result;

    while (currentCount >
           static_cast<std::size_t>(
               topK
           ))
    {
        if (currentCount >
            static_cast<std::size_t>(
                INT_MAX
            ))
        {
            throw std::overflow_error(
                "Top-K candidate count exceeds int range."
            );
        }

        const std::size_t blockCount =
            ceilDivide(
                currentCount,
                static_cast<std::size_t>(
                    TOPK_BLOCK_SIZE
                )
            );

        topKFromCandidatesKernel<<<
            static_cast<unsigned int>(
                blockCount
            ),
            TOPK_BLOCK_SIZE
        >>>(
            currentBuffer,
            nextBuffer,
            static_cast<int>(
                currentCount
            ),
            topK
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        currentCount =
            blockCount *
            static_cast<std::size_t>(
                topK
            );

        std::swap(
            currentBuffer,
            nextBuffer
        );

        ++result.mergePasses;
    }

    result.finalBuffer =
        currentBuffer;

    return result;
}

} // namespace

// =========================================================
// Constructor
// =========================================================

GpuVectorIndex::GpuVectorIndex(
    const std::vector<float>& database,
    const int vectorCount,
    const int dimension)
    : vectorCount_(
        vectorCount
    ),
      dimension_(
        dimension
    ),
      deviceDatabase_(
        validateDatabase(
            database,
            vectorCount,
            dimension
        )
    ),
      deviceQuery_(
        requirePositive(
            dimension,
            "dimension"
        )
    ),
      deviceDistances_(
        requirePositive(
            vectorCount,
            "vectorCount"
        )
    ),
      deviceTopKBufferA_(
        calculateTopKBufferCapacity(
            vectorCount
        )
    ),
      deviceTopKBufferB_(
        calculateTopKBufferCapacity(
            vectorCount
        )
    )
{
    CUDA_CHECK(
        cudaEventCreate(
            &distanceStart_
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &distanceStop_
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &topKStart_
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &topKStop_
        )
    );

    const std::size_t elementCount =
        static_cast<std::size_t>(
            vectorCount_
        ) *
        static_cast<std::size_t>(
            dimension_
        );

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceDatabase_.data(),
            database.data(),
            elementCount *
                sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    const auto uploadEnd =
        std::chrono::steady_clock::now();

    indexUploadTimeMs_ =
        std::chrono::duration<
            double,
            std::milli
        >(
            uploadEnd -
            uploadStart
        ).count();
}

// =========================================================
// Destructor
// =========================================================

GpuVectorIndex::~GpuVectorIndex()
{
    if (distanceStart_ != nullptr)
    {
        cudaEventDestroy(
            distanceStart_
        );
    }

    if (distanceStop_ != nullptr)
    {
        cudaEventDestroy(
            distanceStop_
        );
    }

    if (topKStart_ != nullptr)
    {
        cudaEventDestroy(
            topKStart_
        );
    }

    if (topKStop_ != nullptr)
    {
        cudaEventDestroy(
            topKStop_
        );
    }
}

// =========================================================
// Configuration validation
// =========================================================

bool GpuVectorIndex::isSupportedBlockSize(
    const int blockSize) noexcept
{
    return
        blockSize == 32 ||
        blockSize == 64 ||
        blockSize == 128 ||
        blockSize == 256;
}

// =========================================================
// Full-distance reference search
// =========================================================

GpuSearchResult GpuVectorIndex::search(
    const std::vector<float>& query,
    const GpuSearchConfig& config)
{
    if (query.size() !=
        static_cast<std::size_t>(
            dimension_
        ))
    {
        throw std::invalid_argument(
            "Query dimension does not match "
            "index dimension."
        );
    }

    if (!isSupportedBlockSize(
            config.blockSize
        ))
    {
        throw std::invalid_argument(
            "Unsupported CUDA block size."
        );
    }

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuSearchResult result;

    result.distances.resize(
        static_cast<std::size_t>(
            vectorCount_
        )
    );

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceQuery_.data(),
            query.data(),
            static_cast<std::size_t>(
                dimension_
            ) *
                sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    const auto uploadEnd =
        std::chrono::steady_clock::now();

    result.queryUploadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            uploadEnd -
            uploadStart
        ).count();

    CUDA_CHECK(
        cudaEventRecord(
            distanceStart_
        )
    );

    launchConfiguredKernel(
        config,
        deviceDatabase_.data(),
        deviceQuery_.data(),
        deviceDistances_.data(),
        vectorCount_,
        dimension_
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaEventRecord(
            distanceStop_
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            distanceStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.kernelTimeMs,
            distanceStart_,
            distanceStop_
        )
    );

    const auto downloadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            result.distances.data(),
            deviceDistances_.data(),
            static_cast<std::size_t>(
                vectorCount_
            ) *
                sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    const auto downloadEnd =
        std::chrono::steady_clock::now();

    result.resultDownloadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            downloadEnd -
            downloadStart
        ).count();

    const auto totalEnd =
        std::chrono::steady_clock::now();

    result.totalTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            totalEnd -
            totalStart
        ).count();

    return result;
}

// =========================================================
// Milestone 5 baseline Top-K
// =========================================================

GpuTopKResult GpuVectorIndex::searchTopK(
    const std::vector<float>& query,
    const int topK,
    const GpuSearchConfig& config)
{
    if (query.size() !=
        static_cast<std::size_t>(
            dimension_
        ))
    {
        throw std::invalid_argument(
            "Query dimension does not match "
            "index dimension."
        );
    }

    if (!isSupportedBlockSize(
            config.blockSize
        ))
    {
        throw std::invalid_argument(
            "Unsupported CUDA block size."
        );
    }

    if (topK <= 0 ||
        topK > MAX_TOP_K ||
        topK > vectorCount_)
    {
        throw std::invalid_argument(
            "topK must be between 1 and "
            "min(vectorCount, 32)."
        );
    }

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuTopKResult result;

    result.neighbors.resize(
        static_cast<std::size_t>(
            topK
        )
    );

    // -----------------------------------------------------
    // Query upload
    // -----------------------------------------------------

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceQuery_.data(),
            query.data(),
            static_cast<std::size_t>(
                dimension_
            ) *
                sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    const auto uploadEnd =
        std::chrono::steady_clock::now();

    result.queryUploadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            uploadEnd -
            uploadStart
        ).count();

    // -----------------------------------------------------
    // Full distance kernel
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            distanceStart_
        )
    );

    launchConfiguredKernel(
        config,
        deviceDatabase_.data(),
        deviceQuery_.data(),
        deviceDistances_.data(),
        vectorCount_,
        dimension_
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaEventRecord(
            distanceStop_
        )
    );

    // -----------------------------------------------------
    // Top-K first stage
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            topKStart_
        )
    );

    const std::size_t firstBlockCount =
        ceilDivide(
            static_cast<std::size_t>(
                vectorCount_
            ),
            static_cast<std::size_t>(
                TOPK_BLOCK_SIZE
            )
        );

    topKFromDistancesKernel<<<
        static_cast<unsigned int>(
            firstBlockCount
        ),
        TOPK_BLOCK_SIZE
    >>>(
        deviceDistances_.data(),
        deviceTopKBufferA_.data(),
        vectorCount_,
        topK
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    const std::size_t firstCandidateCount =
        firstBlockCount *
        static_cast<std::size_t>(
            topK
        );

    const CandidateMergeResult merge =
        mergeTopKCandidates(
            deviceTopKBufferA_.data(),
            deviceTopKBufferB_.data(),
            firstCandidateCount,
            topK
        );

    result.topKPasses =
        1 +
        merge.mergePasses;

    CUDA_CHECK(
        cudaEventRecord(
            topKStop_
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            topKStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.distanceKernelTimeMs,
            distanceStart_,
            distanceStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.topKKernelTimeMs,
            topKStart_,
            topKStop_
        )
    );

    // -----------------------------------------------------
    // Download only K candidates
    // -----------------------------------------------------

    const auto downloadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            result.neighbors.data(),
            merge.finalBuffer,
            static_cast<std::size_t>(
                topK
            ) *
                sizeof(Neighbor),
            cudaMemcpyDeviceToHost
        )
    );

    const auto downloadEnd =
        std::chrono::steady_clock::now();

    result.resultDownloadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            downloadEnd -
            downloadStart
        ).count();

    const auto totalEnd =
        std::chrono::steady_clock::now();

    result.totalTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            totalEnd -
            totalStart
        ).count();

    return result;
}

// =========================================================
// Milestone 6 fused search
// =========================================================

GpuFusedTopKResult
GpuVectorIndex::searchTopKFused(
    const std::vector<float>& query,
    const int topK)
{
    if (query.size() !=
        static_cast<std::size_t>(
            dimension_
        ))
    {
        throw std::invalid_argument(
            "Query dimension does not match "
            "index dimension."
        );
    }

    if (topK <= 0 ||
        topK > MAX_TOP_K ||
        topK > vectorCount_)
    {
        throw std::invalid_argument(
            "topK must be between 1 and "
            "min(vectorCount, 32)."
        );
    }

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuFusedTopKResult result;

    result.neighbors.resize(
        static_cast<std::size_t>(
            topK
        )
    );

    // -----------------------------------------------------
    // Query upload
    // -----------------------------------------------------

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceQuery_.data(),
            query.data(),
            static_cast<std::size_t>(
                dimension_
            ) *
                sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    const auto uploadEnd =
        std::chrono::steady_clock::now();

    result.queryUploadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            uploadEnd -
            uploadStart
        ).count();

    // -----------------------------------------------------
    // Fused distance + local Top-K
    // -----------------------------------------------------

    const std::size_t firstBlockCount =
        ceilDivide(
            static_cast<std::size_t>(
                vectorCount_
            ),
            static_cast<std::size_t>(
                FUSED_GROUP_SIZE
            )
        );

    CUDA_CHECK(
        cudaEventRecord(
            distanceStart_
        )
    );

    fusedDistanceLocalTopKKernel<<<
        static_cast<unsigned int>(
            firstBlockCount
        ),
        TOPK_BLOCK_SIZE
    >>>(
        deviceDatabase_.data(),
        deviceQuery_.data(),
        deviceTopKBufferA_.data(),
        vectorCount_,
        dimension_,
        topK
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaEventRecord(
            distanceStop_
        )
    );

    // -----------------------------------------------------
    // Hierarchical candidate-only merge
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            topKStart_
        )
    );

    const std::size_t firstCandidateCount =
        firstBlockCount *
        static_cast<std::size_t>(
            topK
        );

    const CandidateMergeResult merge =
        mergeTopKCandidates(
            deviceTopKBufferA_.data(),
            deviceTopKBufferB_.data(),
            firstCandidateCount,
            topK
        );

    result.topKPasses =
        1 +
        merge.mergePasses;

    CUDA_CHECK(
        cudaEventRecord(
            topKStop_
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            topKStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.fusedKernelTimeMs,
            distanceStart_,
            distanceStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.mergeKernelTimeMs,
            topKStart_,
            topKStop_
        )
    );

    // -----------------------------------------------------
    // Download only K neighbors
    // -----------------------------------------------------

    const auto downloadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            result.neighbors.data(),
            merge.finalBuffer,
            static_cast<std::size_t>(
                topK
            ) *
                sizeof(Neighbor),
            cudaMemcpyDeviceToHost
        )
    );

    const auto downloadEnd =
        std::chrono::steady_clock::now();

    result.resultDownloadTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            downloadEnd -
            downloadStart
        ).count();

    const auto totalEnd =
        std::chrono::steady_clock::now();

    result.totalTimeMs =
        std::chrono::duration<
            double,
            std::milli
        >(
            totalEnd -
            totalStart
        ).count();

    return result;
}

// =========================================================
// Full-distance warm-up
// =========================================================

void GpuVectorIndex::warmUp(
    const GpuSearchConfig& config,
    const int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Warm-up iterations must be positive."
        );
    }

    std::vector<float> query(
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        (void)search(
            query,
            config
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}

// =========================================================
// Baseline Top-K warm-up
// =========================================================

void GpuVectorIndex::warmUpTopK(
    const GpuSearchConfig& config,
    const int topK,
    const int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Warm-up iterations must be positive."
        );
    }

    std::vector<float> query(
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        (void)searchTopK(
            query,
            topK,
            config
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}

// =========================================================
// Fused Top-K warm-up
// =========================================================

void GpuVectorIndex::warmUpFusedTopK(
    const int topK,
    const int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Warm-up iterations must be positive."
        );
    }

    std::vector<float> query(
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        (void)searchTopKFused(
            query,
            topK
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}