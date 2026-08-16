#include "gpu_vector_index.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int CUDA_BLOCK_SIZE = 256;

// ---------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------

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
        vectors * dimensions;

    if (database.size() != expectedElements)
    {
        throw std::invalid_argument(
            "Database size does not match "
            "vectorCount * dimension."
        );
    }

    return expectedElements;
}

} // namespace

// ---------------------------------------------------------
// CUDA L2 distance kernel
// ---------------------------------------------------------

__global__ void l2DistanceKernel(
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

    if (vectorIndex >= vectorCount)
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

    for (int d = threadId;
         d < dimension;
         d += static_cast<int>(
             blockDim.x
         ))
    {
        const float difference =
            vector[d] -
            query[d];

        localSum +=
            difference * difference;
    }

    partial[threadId] =
        localSum;

    __syncthreads();

    // -----------------------------------------------------
    // Shared-memory parallel reduction
    // -----------------------------------------------------

    for (unsigned int stride =
             blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            partial[threadIdx.x] +=
                partial[
                    threadIdx.x +
                    stride
                ];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        distances[vectorIndex] =
            partial[0];
    }
}

// ---------------------------------------------------------
// Constructor
// ---------------------------------------------------------

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
    )
{
    // -----------------------------------------------------
    // Persistent CUDA timing events
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventCreate(
            &kernelStart_
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &kernelStop_
        )
    );

    // -----------------------------------------------------
    // Upload database once
    // -----------------------------------------------------

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

// ---------------------------------------------------------
// Destructor
// ---------------------------------------------------------

GpuVectorIndex::~GpuVectorIndex()
{
    if (kernelStart_ != nullptr)
    {
        cudaEventDestroy(
            kernelStart_
        );
    }

    if (kernelStop_ != nullptr)
    {
        cudaEventDestroy(
            kernelStop_
        );
    }
}

// ---------------------------------------------------------
// Search
// ---------------------------------------------------------

GpuSearchResult GpuVectorIndex::search(
    const std::vector<float>& query)
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

    // End-to-end query timing starts BEFORE
    // allocating the host result vector.
    const auto totalStart =
        std::chrono::steady_clock::now();

    GpuSearchResult result;

    result.distances.resize(
        static_cast<std::size_t>(
            vectorCount_
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
    // Execute CUDA kernel
    // -----------------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            kernelStart_
        )
    );

    const std::size_t sharedMemoryBytes =
        static_cast<std::size_t>(
            CUDA_BLOCK_SIZE
        ) *
        sizeof(float);

    l2DistanceKernel<<<
        vectorCount_,
        CUDA_BLOCK_SIZE,
        sharedMemoryBytes
    >>>(
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
            kernelStop_
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            kernelStop_
        )
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &result.kernelTimeMs,
            kernelStart_,
            kernelStop_
        )
    );

    // -----------------------------------------------------
    // Download all distances
    // -----------------------------------------------------

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

    // -----------------------------------------------------
    // End-to-end query latency
    // -----------------------------------------------------

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

// ---------------------------------------------------------
// Warm-up
// ---------------------------------------------------------

void GpuVectorIndex::warmUp()
{
    std::vector<float> warmupQuery(
        static_cast<std::size_t>(
            dimension_
        ),
        0.0F
    );

    (void)search(
        warmupQuery
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}