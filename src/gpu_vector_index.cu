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

    return static_cast<std::size_t>(
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
        >::max() / dimensions)
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
// GPU Top-K helpers
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
         sequenceSize <= BLOCK_SIZE;
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
// Top-K stage 1
//
// Input:
//   full distance array
//
// Output:
//   K best candidates per 256-distance block
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
// Top-K merge stage
//
// Input:
//   candidates from previous hierarchy level
//
// Output:
//   K best candidates per 256-candidate block
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
// Distance-kernel launch helpers
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
// Block-size validation
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
// Hierarchical GPU Top-K search
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
    // Upload query
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
    // Distance kernel
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
    // Hierarchical Top-K
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

    result.topKPasses =
        1;

    std::size_t currentCount =
        firstBlockCount *
        static_cast<std::size_t>(
            topK
        );

    Neighbor* currentBuffer =
        deviceTopKBufferA_.data();

    Neighbor* nextBuffer =
        deviceTopKBufferB_.data();

    while (currentCount >
           static_cast<std::size_t>(
               topK
           ))
    {
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

        ++result.topKPasses;
    }

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
            currentBuffer,
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
// Top-K warm-up
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