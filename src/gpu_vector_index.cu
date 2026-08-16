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

std::size_t checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    const char* errorMessage)
{
    if (left != 0 &&
        right >
            std::numeric_limits<
                std::size_t
            >::max() /
            left)
    {
        throw std::overflow_error(
            errorMessage
        );
    }

    return
        left *
        right;
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

    const std::size_t expectedElements =
        checkedMultiply(
            vectors,
            dimensions,
            "Database dimensions overflow size_t."
        );

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

    return
        checkedMultiply(
            blockCount,
            static_cast<std::size_t>(
                GpuVectorIndex::MAX_TOP_K
            ),
            "Top-K intermediate buffer size overflow."
        );
}

std::size_t calculateBatchTopKBufferCapacity(
    const int vectorCount)
{
    const std::size_t perQueryCapacity =
        calculateTopKBufferCapacity(
            vectorCount
        );

    return
        checkedMultiply(
            perQueryCapacity,
            static_cast<std::size_t>(
                GpuVectorIndex::MAX_BATCH_SIZE
            ),
            "Batch Top-K buffer size overflow."
        );
}

std::size_t calculateBatchQueryCapacity(
    const int dimension)
{
    return
        checkedMultiply(
            requirePositive(
                dimension,
                "dimension"
            ),
            static_cast<std::size_t>(
                GpuVectorIndex::MAX_BATCH_SIZE
            ),
            "Batch query buffer size overflow."
        );
}

void validateTopK(
    const int topK,
    const int vectorCount)
{
    if (topK <= 0 ||
        topK >
            GpuVectorIndex::MAX_TOP_K ||
        topK > vectorCount)
    {
        throw std::invalid_argument(
            "topK must be between 1 and "
            "min(vectorCount, 32)."
        );
    }
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
// Distance kernels
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
// Milestone 5 first-stage Top-K
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
// Single-query candidate merge
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
// Milestone 6 fused single-query kernel
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
// Milestone 7 fused batch kernel
//
// blockIdx.x -> vector partition
// blockIdx.y -> query
//
// Each block evaluates 256 database vectors against one
// query and emits block-local Top-K candidates.
// =========================================================

__global__ void fusedDistanceLocalTopKBatchKernel(
    const float* __restrict__ database,
    const float* __restrict__ queries,
    Neighbor* __restrict__ output,
    const int vectorCount,
    const int dimension,
    const int topK,
    const std::size_t candidateCapacityPerQuery)
{
    constexpr int WARPS_PER_BLOCK =
        TOPK_BLOCK_SIZE /
        CUDA_WARP_SIZE;

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

    const int queryIndex =
        static_cast<int>(
            blockIdx.y
        );

    const float* query =
        queries +
        static_cast<std::size_t>(
            queryIndex
        ) *
        static_cast<std::size_t>(
            dimension
        );

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

    __syncthreads();

    bitonicSortNeighbors<
        TOPK_BLOCK_SIZE
    >(
        candidates
    );

    if (threadId <
        topK)
    {
        const std::size_t queryOffset =
            static_cast<std::size_t>(
                queryIndex
            ) *
            candidateCapacityPerQuery;

        const std::size_t outputIndex =
            queryOffset +
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
// Milestone 8 query-tiled fused batch kernel
//
// A block evaluates one 256-vector partition against a
// compile-time query tile.
//
// Database values are loaded once and reused across every
// query in the tile.
//
// Supported query-tile sizes:
//   1
//   2
//   4
//
// For QUERY_TILE_SIZE = 4:
//
//   blockIdx.x -> vector partition
//   blockIdx.y -> group of four queries
//
// The 256 block threads are also partitioned into four
// 64-thread cooperative sorting groups, allowing the four
// local 256-candidate Top-K sorts to progress concurrently.
// =========================================================

template <int QUERY_TILE_SIZE>
__global__ void queryTiledDistanceLocalTopKBatchKernel(
    const float* __restrict__ database,
    const float* __restrict__ queries,
    Neighbor* __restrict__ output,
    const int vectorCount,
    const int dimension,
    const int queryCount,
    const int topK,
    const std::size_t candidateCapacityPerQuery)
{
    constexpr int WARPS_PER_BLOCK =
        TOPK_BLOCK_SIZE /
        CUDA_WARP_SIZE;

    static_assert(
        QUERY_TILE_SIZE == 1 ||
        QUERY_TILE_SIZE == 2 ||
        QUERY_TILE_SIZE == 4
    );

    static_assert(
        TOPK_BLOCK_SIZE %
        QUERY_TILE_SIZE == 0
    );

    constexpr int SORT_THREADS_PER_QUERY =
        TOPK_BLOCK_SIZE /
        QUERY_TILE_SIZE;

    static_assert(
        GpuVectorIndex::MAX_TOP_K <=
        SORT_THREADS_PER_QUERY
    );

    __shared__ Neighbor candidates[
        QUERY_TILE_SIZE *
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

    const int queryBase =
        static_cast<int>(
            blockIdx.y
        ) *
        QUERY_TILE_SIZE;

    int activeQueries =
        queryCount -
        queryBase;

    if (activeQueries >
        QUERY_TILE_SIZE)
    {
        activeQueries =
            QUERY_TILE_SIZE;
    }

    // -----------------------------------------------------
    // Initialize every query-local candidate array.
    // -----------------------------------------------------

    for (int candidateIndex =
             threadId;
         candidateIndex <
             QUERY_TILE_SIZE *
             FUSED_GROUP_SIZE;
         candidateIndex +=
             TOPK_BLOCK_SIZE)
    {
        candidates[
            candidateIndex
        ] =
            makeInvalidNeighbor();
    }

    __syncthreads();

    const std::size_t blockVectorBase =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        static_cast<std::size_t>(
            FUSED_GROUP_SIZE
        );

    // -----------------------------------------------------
    // Eight warps cooperate on 256 vectors.
    //
    // For every vector component we load the database value
    // once, then reuse it across all active queries.
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

        float localSums[
            QUERY_TILE_SIZE
        ];

        #pragma unroll
        for (int querySlot = 0;
             querySlot <
                 QUERY_TILE_SIZE;
             ++querySlot)
        {
            localSums[
                querySlot
            ] =
                0.0F;
        }

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

            for (int dimensionIndex =
                     laneId;
                 dimensionIndex <
                     dimension;
                 dimensionIndex +=
                     CUDA_WARP_SIZE)
            {
                const float vectorValue =
                    vector[
                        dimensionIndex
                    ];

                #pragma unroll
                for (int querySlot = 0;
                     querySlot <
                         QUERY_TILE_SIZE;
                     ++querySlot)
                {
                    if (querySlot <
                        activeQueries)
                    {
                        const int queryIndex =
                            queryBase +
                            querySlot;

                        const float queryValue =
                            queries[
                                static_cast<
                                    std::size_t
                                >(
                                    queryIndex
                                ) *
                                static_cast<
                                    std::size_t
                                >(
                                    dimension
                                ) +
                                static_cast<
                                    std::size_t
                                >(
                                    dimensionIndex
                                )
                            ];

                        const float difference =
                            vectorValue -
                            queryValue;

                        localSums[
                            querySlot
                        ] +=
                            difference *
                            difference;
                    }
                }
            }
        }

        #pragma unroll
        for (int querySlot = 0;
             querySlot <
                 QUERY_TILE_SIZE;
             ++querySlot)
        {
            const float distance =
                warpReduceSum(
                    localSums[
                        querySlot
                    ]
                );

            if (laneId == 0 &&
                querySlot <
                    activeQueries &&
                vectorIndex <
                    static_cast<
                        std::size_t
                    >(
                        vectorCount
                    ))
            {
                candidates[
                    querySlot *
                    FUSED_GROUP_SIZE +
                    localVectorIndex
                ] =
                    Neighbor{
                        static_cast<int>(
                            vectorIndex
                        ),
                        distance
                    };
            }
        }
    }

    __syncthreads();

    // -----------------------------------------------------
    // Sort all query-local candidate arrays concurrently.
    //
    // QUERY_TILE_SIZE = 4:
    //   64 threads/query
    //
    // Each sorting thread owns multiple candidate indices.
    // Every block thread reaches every synchronization point.
    // -----------------------------------------------------

    const int sortQuerySlot =
        threadId /
        SORT_THREADS_PER_QUERY;

    const int sortThreadId =
        threadId %
        SORT_THREADS_PER_QUERY;

    Neighbor* sortValues =
        candidates +
        sortQuerySlot *
        FUSED_GROUP_SIZE;

    for (unsigned int sequenceSize = 2;
         sequenceSize <=
             TOPK_BLOCK_SIZE;
         sequenceSize <<= 1)
    {
        for (unsigned int stride =
                 sequenceSize >> 1;
             stride > 0;
             stride >>= 1)
        {
            for (int candidateIndex =
                     sortThreadId;
                 candidateIndex <
                     TOPK_BLOCK_SIZE;
                 candidateIndex +=
                     SORT_THREADS_PER_QUERY)
            {
                const unsigned int partner =
                    static_cast<
                        unsigned int
                    >(
                        candidateIndex
                    ) ^
                    stride;

                if (partner >
                    static_cast<
                        unsigned int
                    >(
                        candidateIndex
                    ))
                {
                    const bool ascending =
                        (
                            static_cast<
                                unsigned int
                            >(
                                candidateIndex
                            ) &
                            sequenceSize
                        ) == 0;

                    const Neighbor left =
                        sortValues[
                            candidateIndex
                        ];

                    const Neighbor right =
                        sortValues[
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
                        sortValues[
                            candidateIndex
                        ] =
                            right;

                        sortValues[
                            partner
                        ] =
                            left;
                    }
                }
            }

            __syncthreads();
        }
    }

    // -----------------------------------------------------
    // Emit K candidates for every active query.
    // -----------------------------------------------------

    if (sortQuerySlot <
            activeQueries &&
        sortThreadId <
            topK)
    {
        const int queryIndex =
            queryBase +
            sortQuerySlot;

        const std::size_t queryOffset =
            static_cast<std::size_t>(
                queryIndex
            ) *
            candidateCapacityPerQuery;

        const std::size_t outputIndex =
            queryOffset +
            static_cast<std::size_t>(
                blockIdx.x
            ) *
            static_cast<std::size_t>(
                topK
            ) +
            static_cast<std::size_t>(
                sortThreadId
            );

        output[
            outputIndex
        ] =
            sortValues[
                sortThreadId
            ];
    }
}

// =========================================================
// Milestone 7 batched candidate merge
// =========================================================

__global__ void topKFromCandidatesBatchKernel(
    const Neighbor* __restrict__ input,
    Neighbor* __restrict__ output,
    const int countPerQuery,
    const int topK,
    const std::size_t candidateCapacityPerQuery)
{
    __shared__ Neighbor candidates[
        TOPK_BLOCK_SIZE
    ];

    const int threadId =
        static_cast<int>(
            threadIdx.x
        );

    const int queryIndex =
        static_cast<int>(
            blockIdx.y
        );

    const std::size_t queryOffset =
        static_cast<std::size_t>(
            queryIndex
        ) *
        candidateCapacityPerQuery;

    const std::size_t localIndex =
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

    if (localIndex <
        static_cast<std::size_t>(
            countPerQuery
        ))
    {
        candidate =
            input[
                queryOffset +
                localIndex
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
            queryOffset +
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
// Host launch helpers
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


int selectQueryTileSize(
    const int queryCount)
{
    if (queryCount >= 4)
    {
        return 4;
    }

    if (queryCount >= 2)
    {
        return 2;
    }

    return 1;
}

template <int QUERY_TILE_SIZE>
void launchQueryTiledBatchKernel(
    const float* database,
    const float* queries,
    Neighbor* output,
    const int vectorCount,
    const int dimension,
    const int queryCount,
    const int topK,
    const std::size_t candidateCapacityPerQuery,
    const std::size_t firstBlockCount)
{
    const std::size_t queryTileCount =
        ceilDivide(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                QUERY_TILE_SIZE
            )
        );

    const dim3 grid(
        static_cast<unsigned int>(
            firstBlockCount
        ),
        static_cast<unsigned int>(
            queryTileCount
        ),
        1U
    );

    queryTiledDistanceLocalTopKBatchKernel<
        QUERY_TILE_SIZE
    ><<<
        grid,
        TOPK_BLOCK_SIZE
    >>>(
        database,
        queries,
        output,
        vectorCount,
        dimension,
        queryCount,
        topK,
        candidateCapacityPerQuery
    );
}

void launchAdaptiveQueryTiledBatchKernel(
    const int queryTileSize,
    const float* database,
    const float* queries,
    Neighbor* output,
    const int vectorCount,
    const int dimension,
    const int queryCount,
    const int topK,
    const std::size_t candidateCapacityPerQuery,
    const std::size_t firstBlockCount)
{
    switch (queryTileSize)
    {
        case 1:
            launchQueryTiledBatchKernel<1>(
                database,
                queries,
                output,
                vectorCount,
                dimension,
                queryCount,
                topK,
                candidateCapacityPerQuery,
                firstBlockCount
            );
            break;

        case 2:
            launchQueryTiledBatchKernel<2>(
                database,
                queries,
                output,
                vectorCount,
                dimension,
                queryCount,
                topK,
                candidateCapacityPerQuery,
                firstBlockCount
            );
            break;

        case 4:
            launchQueryTiledBatchKernel<4>(
                database,
                queries,
                output,
                vectorCount,
                dimension,
                queryCount,
                topK,
                candidateCapacityPerQuery,
                firstBlockCount
            );
            break;

        default:
            throw std::invalid_argument(
                "Unsupported query tile size."
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

struct BatchCandidateMergeResult
{
    Neighbor* finalBuffer = nullptr;
    int mergePasses = 0;
};

BatchCandidateMergeResult
mergeTopKCandidatesBatch(
    Neighbor* currentBuffer,
    Neighbor* nextBuffer,
    std::size_t currentCountPerQuery,
    const int queryCount,
    const int topK,
    const std::size_t candidateCapacityPerQuery)
{
    BatchCandidateMergeResult result;

    while (currentCountPerQuery >
           static_cast<std::size_t>(
               topK
           ))
    {
        if (currentCountPerQuery >
            static_cast<std::size_t>(
                INT_MAX
            ))
        {
            throw std::overflow_error(
                "Batch candidate count exceeds int range."
            );
        }

        const std::size_t blockCount =
            ceilDivide(
                currentCountPerQuery,
                static_cast<std::size_t>(
                    TOPK_BLOCK_SIZE
                )
            );

        const dim3 grid(
            static_cast<unsigned int>(
                blockCount
            ),
            static_cast<unsigned int>(
                queryCount
            ),
            1U
        );

        topKFromCandidatesBatchKernel<<<
            grid,
            TOPK_BLOCK_SIZE
        >>>(
            currentBuffer,
            nextBuffer,
            static_cast<int>(
                currentCountPerQuery
            ),
            topK,
            candidateCapacityPerQuery
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        currentCountPerQuery =
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
    ),
      deviceBatchQueries_(
        calculateBatchQueryCapacity(
            dimension
        )
    ),
      deviceBatchTopKBufferA_(
        calculateBatchTopKBufferCapacity(
            vectorCount
        )
    ),
      deviceBatchTopKBufferB_(
        calculateBatchTopKBufferCapacity(
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
// Milestone 5 baseline
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

    validateTopK(
        topK,
        vectorCount_
    );

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuTopKResult result;

    result.neighbors.resize(
        static_cast<std::size_t>(
            topK
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

    const CandidateMergeResult merge =
        mergeTopKCandidates(
            deviceTopKBufferA_.data(),
            deviceTopKBufferB_.data(),
            firstBlockCount *
                static_cast<std::size_t>(
                    topK
                ),
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
// Milestone 6 single-query fused search
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

    validateTopK(
        topK,
        vectorCount_
    );

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuFusedTopKResult result;

    result.neighbors.resize(
        static_cast<std::size_t>(
            topK
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

    CUDA_CHECK(
        cudaEventRecord(
            topKStart_
        )
    );

    const CandidateMergeResult merge =
        mergeTopKCandidates(
            deviceTopKBufferA_.data(),
            deviceTopKBufferB_.data(),
            firstBlockCount *
                static_cast<std::size_t>(
                    topK
                ),
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
// Milestone 7 batch search
// =========================================================

GpuBatchTopKResult
GpuVectorIndex::searchTopKBatch(
    const std::vector<float>& queries,
    const int queryCount,
    const int topK)
{
    if (queryCount <= 0 ||
        queryCount >
            MAX_BATCH_SIZE)
    {
        throw std::invalid_argument(
            "queryCount must be between "
            "1 and 32."
        );
    }

    validateTopK(
        topK,
        vectorCount_
    );

    const std::size_t expectedElements =
        checkedMultiply(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                dimension_
            ),
            "Batch query dimensions overflow size_t."
        );

    if (queries.size() !=
        expectedElements)
    {
        throw std::invalid_argument(
            "queries.size() must equal "
            "queryCount * dimension."
        );
    }

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuBatchTopKResult result;

    result.queryCount =
        queryCount;

    result.topK =
        topK;

    result.queryTileSize =
        1;

    result.neighbors.resize(
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            topK
        )
    );

    // -----------------------------------------------------
    // Upload all queries in one transfer.
    // -----------------------------------------------------

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceBatchQueries_.data(),
            queries.data(),
            expectedElements *
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

    const std::size_t firstBlockCount =
        ceilDivide(
            static_cast<std::size_t>(
                vectorCount_
            ),
            static_cast<std::size_t>(
                FUSED_GROUP_SIZE
            )
        );

    const std::size_t
        candidateCapacityPerQuery =
            calculateTopKBufferCapacity(
                vectorCount_
            );

    // -----------------------------------------------------
    // 2D fused search:
    //
    // x = database partition
    // y = query
    // -----------------------------------------------------

    const dim3 fusedGrid(
        static_cast<unsigned int>(
            firstBlockCount
        ),
        static_cast<unsigned int>(
            queryCount
        ),
        1U
    );

    CUDA_CHECK(
        cudaEventRecord(
            distanceStart_
        )
    );

    fusedDistanceLocalTopKBatchKernel<<<
        fusedGrid,
        TOPK_BLOCK_SIZE
    >>>(
        deviceDatabase_.data(),
        deviceBatchQueries_.data(),
        deviceBatchTopKBufferA_.data(),
        vectorCount_,
        dimension_,
        topK,
        candidateCapacityPerQuery
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
    // Per-query hierarchical merge.
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            topKStart_
        )
    );

    const std::size_t
        firstCandidateCountPerQuery =
            firstBlockCount *
            static_cast<std::size_t>(
                topK
            );

    const BatchCandidateMergeResult merge =
        mergeTopKCandidatesBatch(
            deviceBatchTopKBufferA_.data(),
            deviceBatchTopKBufferB_.data(),
            firstCandidateCountPerQuery,
            queryCount,
            topK,
            candidateCapacityPerQuery
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
    // Each query occupies a fixed candidate segment on GPU.
    //
    // cudaMemcpy2D gathers only the first K neighbors from
    // each segment into one contiguous host result vector.
    // -----------------------------------------------------

    const auto downloadStart =
        std::chrono::steady_clock::now();

    const std::size_t copyWidth =
        static_cast<std::size_t>(
            topK
        ) *
        sizeof(Neighbor);

    const std::size_t sourcePitch =
        candidateCapacityPerQuery *
        sizeof(Neighbor);

    CUDA_CHECK(
        cudaMemcpy2D(
            result.neighbors.data(),
            copyWidth,
            merge.finalBuffer,
            sourcePitch,
            copyWidth,
            static_cast<std::size_t>(
                queryCount
            ),
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
// Milestone 8 query-tiled batch search
// =========================================================

GpuBatchTopKResult
GpuVectorIndex::searchTopKBatchTiled(
    const std::vector<float>& queries,
    const int queryCount,
    const int topK)
{
    if (queryCount <= 0 ||
        queryCount >
            MAX_BATCH_SIZE)
    {
        throw std::invalid_argument(
            "queryCount must be between "
            "1 and 32."
        );
    }

    validateTopK(
        topK,
        vectorCount_
    );

    const std::size_t expectedElements =
        checkedMultiply(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                dimension_
            ),
            "Batch query dimensions overflow size_t."
        );

    if (queries.size() !=
        expectedElements)
    {
        throw std::invalid_argument(
            "queries.size() must equal "
            "queryCount * dimension."
        );
    }

    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuBatchTopKResult result;

    result.queryCount =
        queryCount;

    result.topK =
        topK;

    result.queryTileSize =
        selectQueryTileSize(
            queryCount
        );

    result.neighbors.resize(
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            topK
        )
    );

    // -----------------------------------------------------
    // Upload the query batch once.
    // -----------------------------------------------------

    const auto uploadStart =
        std::chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMemcpy(
            deviceBatchQueries_.data(),
            queries.data(),
            expectedElements *
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

    const std::size_t firstBlockCount =
        ceilDivide(
            static_cast<std::size_t>(
                vectorCount_
            ),
            static_cast<std::size_t>(
                FUSED_GROUP_SIZE
            )
        );

    const std::size_t
        candidateCapacityPerQuery =
            calculateTopKBufferCapacity(
                vectorCount_
            );

    // -----------------------------------------------------
    // Query-tiled fused stage.
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            distanceStart_
        )
    );

    launchAdaptiveQueryTiledBatchKernel(
        result.queryTileSize,
        deviceDatabase_.data(),
        deviceBatchQueries_.data(),
        deviceBatchTopKBufferA_.data(),
        vectorCount_,
        dimension_,
        queryCount,
        topK,
        candidateCapacityPerQuery,
        firstBlockCount
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
    // Reuse the exact Milestone 7 candidate hierarchy.
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            topKStart_
        )
    );

    const std::size_t
        firstCandidateCountPerQuery =
            firstBlockCount *
            static_cast<std::size_t>(
                topK
            );

    const BatchCandidateMergeResult merge =
        mergeTopKCandidatesBatch(
            deviceBatchTopKBufferA_.data(),
            deviceBatchTopKBufferB_.data(),
            firstCandidateCountPerQuery,
            queryCount,
            topK,
            candidateCapacityPerQuery
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
    // Gather K neighbors/query into contiguous host memory.
    // -----------------------------------------------------

    const auto downloadStart =
        std::chrono::steady_clock::now();

    const std::size_t copyWidth =
        static_cast<std::size_t>(
            topK
        ) *
        sizeof(Neighbor);

    const std::size_t sourcePitch =
        candidateCapacityPerQuery *
        sizeof(Neighbor);

    CUDA_CHECK(
        cudaMemcpy2D(
            result.neighbors.data(),
            copyWidth,
            merge.finalBuffer,
            sourcePitch,
            copyWidth,
            static_cast<std::size_t>(
                queryCount
            ),
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
// Warm-up methods
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

void GpuVectorIndex::warmUpTopKBatch(
    const int queryCount,
    const int topK,
    const int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Warm-up iterations must be positive."
        );
    }

    if (queryCount <= 0 ||
        queryCount >
            MAX_BATCH_SIZE)
    {
        throw std::invalid_argument(
            "queryCount must be between "
            "1 and 32."
        );
    }

    std::vector<float> queries(
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        (void)searchTopKBatch(
            queries,
            queryCount,
            topK
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}

void GpuVectorIndex::warmUpTopKBatchTiled(
    const int queryCount,
    const int topK,
    const int iterations)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument(
            "Warm-up iterations must be positive."
        );
    }

    if (queryCount <= 0 ||
        queryCount >
            MAX_BATCH_SIZE)
    {
        throw std::invalid_argument(
            "queryCount must be between "
            "1 and 32."
        );
    }

    std::vector<float> queries(
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration)
    {
        (void)searchTopKBatchTiled(
            queries,
            queryCount,
            topK
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}

