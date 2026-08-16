#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "gpu_vector_index.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr int DEFAULT_VECTOR_COUNT = 100000;
constexpr int DEFAULT_DIMENSION = 128;
constexpr int DEFAULT_TOP_K = 10;

}

// ---------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------

std::vector<float> generateRandomData(
    const std::size_t count,
    const unsigned int seed)
{
    std::mt19937 generator(
        seed
    );

    std::uniform_real_distribution<float>
        distribution(
            -1.0F,
            1.0F
        );

    std::vector<float> values(
        count
    );

    for (float& value : values)
    {
        value =
            distribution(
                generator
            );
    }

    return values;
}

// ---------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------

std::vector<float> calculateDistancesCpu(
    const std::vector<float>& database,
    const std::vector<float>& query,
    const int vectorCount,
    const int dimension)
{
    std::vector<float> distances(
        static_cast<std::size_t>(
            vectorCount
        )
    );

    for (int vectorIndex = 0;
         vectorIndex < vectorCount;
         ++vectorIndex)
    {
        float distance = 0.0F;

        const std::size_t offset =
            static_cast<std::size_t>(
                vectorIndex
            ) *
            static_cast<std::size_t>(
                dimension
            );

        for (int d = 0;
             d < dimension;
             ++d)
        {
            const float difference =
                database[
                    offset +
                    static_cast<std::size_t>(
                        d
                    )
                ] -
                query[
                    static_cast<std::size_t>(
                        d
                    )
                ];

            distance +=
                difference *
                difference;
        }

        distances[
            static_cast<std::size_t>(
                vectorIndex
            )
        ] = distance;
    }

    return distances;
}

// ---------------------------------------------------------
// Correctness validation
// ---------------------------------------------------------

float calculateMaxAbsoluteError(
    const std::vector<float>& expected,
    const std::vector<float>& actual)
{
    if (expected.size() !=
        actual.size())
    {
        return
            std::numeric_limits<
                float
            >::infinity();
    }

    float maxError = 0.0F;

    for (std::size_t index = 0;
         index < expected.size();
         ++index)
    {
        const float error =
            std::abs(
                expected[index] -
                actual[index]
            );

        maxError =
            std::max(
                maxError,
                error
            );
    }

    return maxError;
}

// ---------------------------------------------------------
// CPU Top-K
// ---------------------------------------------------------

std::vector<int> findTopK(
    const std::vector<float>& distances,
    const int k)
{
    std::vector<int> indices(
        distances.size()
    );

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&distances](
            const int left,
            const int right)
        {
            const float leftDistance =
                distances[
                    static_cast<
                        std::size_t
                    >(left)
                ];

            const float rightDistance =
                distances[
                    static_cast<
                        std::size_t
                    >(right)
                ];

            if (leftDistance ==
                rightDistance)
            {
                return left < right;
            }

            return
                leftDistance <
                rightDistance;
        }
    );

    indices.resize(
        static_cast<std::size_t>(
            k
        )
    );

    return indices;
}

// ---------------------------------------------------------
// Print Top-K
// ---------------------------------------------------------

void printTopK(
    const std::vector<int>& indices,
    const std::vector<float>& distances)
{
    std::cout
        << "\nTop-K nearest vectors\n"
        << "-----------------------------\n";

    for (std::size_t rank = 0;
         rank < indices.size();
         ++rank)
    {
        const int index =
            indices[rank];

        std::cout
            << "#"
            << std::setw(2)
            << rank + 1
            << "  vector="
            << std::setw(7)
            << index
            << "  squared_l2="
            << std::fixed
            << std::setprecision(6)
            << distances[
                static_cast<
                    std::size_t
                >(index)
            ]
            << '\n';
    }
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main(
    const int argc,
    char** argv)
{
    try
    {
        int vectorCount =
            DEFAULT_VECTOR_COUNT;

        int dimension =
            DEFAULT_DIMENSION;

        int topK =
            DEFAULT_TOP_K;

        // -------------------------------------------------
        // CLI arguments
        // -------------------------------------------------

        if (argc >= 2)
        {
            vectorCount =
                std::stoi(
                    argv[1]
                );
        }

        if (argc >= 3)
        {
            dimension =
                std::stoi(
                    argv[2]
                );
        }

        if (argc >= 4)
        {
            topK =
                std::stoi(
                    argv[3]
                );
        }

        if (vectorCount <= 0 ||
            dimension <= 0 ||
            topK <= 0 ||
            topK > vectorCount)
        {
            std::cerr
                << "Invalid arguments.\n";

            std::cerr
                << "Usage: vortex_cuda "
                << "[vectorCount] "
                << "[dimension] "
                << "[topK]\n";

            return EXIT_FAILURE;
        }

        // -------------------------------------------------
        // CUDA device discovery
        // -------------------------------------------------

        int deviceCount = 0;

        CUDA_CHECK(
            cudaGetDeviceCount(
                &deviceCount
            )
        );

        if (deviceCount == 0)
        {
            std::cerr
                << "No CUDA-capable GPU detected.\n";

            return EXIT_FAILURE;
        }

        cudaDeviceProp device{};

        CUDA_CHECK(
            cudaGetDeviceProperties(
                &device,
                0
            )
        );

        // -------------------------------------------------
        // Header
        // -------------------------------------------------

        std::cout
            << "====================================\n"
            << "       Vortex CUDA Vector Engine\n"
            << "====================================\n\n";

        std::cout
            << "GPU: "
            << device.name
            << '\n';

        std::cout
            << "Compute capability: "
            << device.major
            << '.'
            << device.minor
            << '\n';

        std::cout
            << "Multiprocessors: "
            << device.multiProcessorCount
            << '\n';

        std::cout
            << "Dataset: "
            << vectorCount
            << " vectors x "
            << dimension
            << " dimensions\n";

        std::cout
            << "Top-K: "
            << topK
            << "\n\n";

        // -------------------------------------------------
        // Dataset generation
        // -------------------------------------------------

        const std::size_t databaseElements =
            static_cast<std::size_t>(
                vectorCount
            ) *
            static_cast<std::size_t>(
                dimension
            );

        std::cout
            << "Generating dataset...\n";

        const std::vector<float> database =
            generateRandomData(
                databaseElements,
                42
            );

        const std::vector<float> query =
            generateRandomData(
                static_cast<std::size_t>(
                    dimension
                ),
                1337
            );

        // -------------------------------------------------
        // CPU benchmark
        // -------------------------------------------------

        std::cout
            << "Running CPU reference implementation...\n";

        const auto cpuStart =
            std::chrono::steady_clock::now();

        const std::vector<float>
            cpuDistances =
                calculateDistancesCpu(
                    database,
                    query,
                    vectorCount,
                    dimension
                );

        const auto cpuEnd =
            std::chrono::steady_clock::now();

        const double cpuTimeMs =
            std::chrono::duration<
                double,
                std::milli
            >(
                cpuEnd -
                cpuStart
            ).count();

        // -------------------------------------------------
        // Persistent GPU index
        // -------------------------------------------------

        std::cout
            << "Building persistent GPU vector index...\n";

        GpuVectorIndex gpuIndex(
            database,
            vectorCount,
            dimension
        );

        const double indexSizeMiB =
            static_cast<double>(
                gpuIndex.indexBytes()
            ) /
            (1024.0 * 1024.0);

        std::cout
            << std::fixed
            << std::setprecision(3);

        std::cout
            << "Index size:          "
            << indexSizeMiB
            << " MiB\n";

        std::cout
            << "One-time upload:     "
            << gpuIndex.indexUploadTimeMs()
            << " ms\n";

        // -------------------------------------------------
        // GPU warm-up
        // -------------------------------------------------

        std::cout
            << "Warming GPU...\n";

        gpuIndex.warmUp();

        // -------------------------------------------------
        // Steady-state GPU query
        // -------------------------------------------------

        std::cout
            << "Running steady-state GPU search...\n";

        const GpuSearchResult gpuResult =
            gpuIndex.search(
                query
            );

        const std::vector<float>&
            gpuDistances =
                gpuResult.distances;

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        const float maxError =
            calculateMaxAbsoluteError(
                cpuDistances,
                gpuDistances
            );

        const std::vector<int> cpuTopK =
            findTopK(
                cpuDistances,
                topK
            );

        const std::vector<int> gpuTopK =
            findTopK(
                gpuDistances,
                topK
            );

        const bool topKMatches =
            cpuTopK ==
            gpuTopK;

        // -------------------------------------------------
        // Benchmark output
        // -------------------------------------------------

        std::cout
            << "\nBenchmark\n"
            << "-----------------------------\n";

        std::cout
            << "CPU computation:    "
            << cpuTimeMs
            << " ms\n";

        std::cout
            << "\nGPU query breakdown\n"
            << "-----------------------------\n";

        std::cout
            << "Query upload:       "
            << gpuResult.queryUploadTimeMs
            << " ms\n";

        std::cout
            << "Kernel:             "
            << gpuResult.kernelTimeMs
            << " ms\n";

        std::cout
            << "Result download:    "
            << gpuResult.resultDownloadTimeMs
            << " ms\n";

        std::cout
            << "GPU query total:    "
            << gpuResult.totalTimeMs
            << " ms\n";

        // -------------------------------------------------
        // Speedups
        // -------------------------------------------------

        if (gpuResult.kernelTimeMs >
            0.0F)
        {
            const double kernelSpeedup =
                cpuTimeMs /
                static_cast<double>(
                    gpuResult.kernelTimeMs
                );

            std::cout
                << "Kernel speedup:     "
                << kernelSpeedup
                << "x\n";
        }

        if (gpuResult.totalTimeMs >
            0.0)
        {
            const double
                endToEndSpeedup =
                    cpuTimeMs /
                    gpuResult.totalTimeMs;

            std::cout
                << "End-to-end speedup: "
                << endToEndSpeedup
                << "x\n";
        }

        // -------------------------------------------------
        // Correctness output
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "-----------------------------\n";

        std::cout
            << std::setprecision(8);

        std::cout
            << "Max abs error:      "
            << maxError
            << '\n';

        std::cout
            << "Top-K match:        "
            << (
                topKMatches
                    ? "YES"
                    : "NO"
            )
            << '\n';

        printTopK(
            gpuTopK,
            gpuDistances
        );

        std::cout
            << "\nMilestone 2 completed successfully.\n";

        return EXIT_SUCCESS;
    }
    catch (
        const std::exception& exception
    )
    {
        std::cerr
            << "Fatal error: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}