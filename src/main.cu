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
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int DEFAULT_VECTOR_COUNT = 100000;
constexpr int DEFAULT_DIMENSION = 128;
constexpr int DEFAULT_TOP_K = 10;

constexpr int WARMUP_RUNS = 5;
constexpr int BENCHMARK_RUNS = 50;

struct TimingStats
{
    double minimum = 0.0;
    double average = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

struct KernelBenchmark
{
    DistanceKernel kernel{};

    TimingStats kernelStats;
    TimingStats totalStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<float> finalDistances;
};

} // namespace

// =========================================================
// Dataset
// =========================================================

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

// =========================================================
// CPU exact distance reference
// =========================================================

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

// =========================================================
// Correctness
// =========================================================

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

// =========================================================
// Top-K
// =========================================================

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
                    static_cast<std::size_t>(
                        left
                    )
                ];

            const float rightDistance =
                distances[
                    static_cast<std::size_t>(
                        right
                    )
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

// =========================================================
// Statistics
// =========================================================

double percentile(
    const std::vector<double>& sortedValues,
    const double percentileValue)
{
    if (sortedValues.empty())
    {
        throw std::invalid_argument(
            "Cannot calculate percentile "
            "of empty data."
        );
    }

    if (sortedValues.size() == 1)
    {
        return sortedValues.front();
    }

    const double position =
        percentileValue *
        static_cast<double>(
            sortedValues.size() - 1
        );

    const std::size_t lowerIndex =
        static_cast<std::size_t>(
            std::floor(position)
        );

    const std::size_t upperIndex =
        static_cast<std::size_t>(
            std::ceil(position)
        );

    if (lowerIndex == upperIndex)
    {
        return
            sortedValues[
                lowerIndex
            ];
    }

    const double fraction =
        position -
        static_cast<double>(
            lowerIndex
        );

    return
        sortedValues[lowerIndex] +
        fraction *
        (
            sortedValues[upperIndex] -
            sortedValues[lowerIndex]
        );
}

TimingStats calculateStats(
    std::vector<double> values)
{
    if (values.empty())
    {
        throw std::invalid_argument(
            "Cannot calculate statistics "
            "from empty values."
        );
    }

    std::sort(
        values.begin(),
        values.end()
    );

    TimingStats stats;

    stats.minimum =
        values.front();

    stats.average =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) /
        static_cast<double>(
            values.size()
        );

    stats.p50 =
        percentile(
            values,
            0.50
        );

    stats.p95 =
        percentile(
            values,
            0.95
        );

    stats.p99 =
        percentile(
            values,
            0.99
        );

    return stats;
}

// =========================================================
// Kernel naming
// =========================================================

const char* kernelName(
    const DistanceKernel kernel)
{
    switch (kernel)
    {
        case DistanceKernel::SharedMemory:
            return "Shared Memory";

        case DistanceKernel::WarpShuffle:
            return "Warp Shuffle";

        default:
            return "Unknown";
    }
}

// =========================================================
// Benchmark one kernel
// =========================================================

KernelBenchmark benchmarkKernel(
    GpuVectorIndex& index,
    const std::vector<float>& query,
    const DistanceKernel kernel)
{
    index.warmUp(
        kernel,
        WARMUP_RUNS
    );

    std::vector<double> kernelTimes;
    std::vector<double> totalTimes;
    std::vector<double> uploadTimes;
    std::vector<double> downloadTimes;

    kernelTimes.reserve(
        BENCHMARK_RUNS
    );

    totalTimes.reserve(
        BENCHMARK_RUNS
    );

    uploadTimes.reserve(
        BENCHMARK_RUNS
    );

    downloadTimes.reserve(
        BENCHMARK_RUNS
    );

    std::vector<float> finalDistances;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuSearchResult result =
            index.search(
                query,
                kernel
            );

        kernelTimes.push_back(
            static_cast<double>(
                result.kernelTimeMs
            )
        );

        totalTimes.push_back(
            result.totalTimeMs
        );

        uploadTimes.push_back(
            result.queryUploadTimeMs
        );

        downloadTimes.push_back(
            result.resultDownloadTimeMs
        );

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            finalDistances =
                std::move(
                    result.distances
                );
        }
    }

    KernelBenchmark benchmark;

    benchmark.kernel =
        kernel;

    benchmark.kernelStats =
        calculateStats(
            kernelTimes
        );

    benchmark.totalStats =
        calculateStats(
            totalTimes
        );

    benchmark.uploadStats =
        calculateStats(
            uploadTimes
        );

    benchmark.downloadStats =
        calculateStats(
            downloadTimes
        );

    benchmark.finalDistances =
        std::move(
            finalDistances
        );

    return benchmark;
}

// =========================================================
// Print benchmark statistics
// =========================================================

void printStats(
    const char* label,
    const TimingStats& stats)
{
    std::cout
        << label
        << '\n';

    std::cout
        << "  Min:       "
        << stats.minimum
        << " ms\n";

    std::cout
        << "  Average:   "
        << stats.average
        << " ms\n";

    std::cout
        << "  P50:       "
        << stats.p50
        << " ms\n";

    std::cout
        << "  P95:       "
        << stats.p95
        << " ms\n";

    std::cout
        << "  P99:       "
        << stats.p99
        << " ms\n";
}

void printKernelBenchmark(
    const KernelBenchmark& benchmark)
{
    std::cout
        << "\n"
        << kernelName(
            benchmark.kernel
        )
        << " Kernel\n";

    std::cout
        << "------------------------------------\n";

    printStats(
        "Kernel latency",
        benchmark.kernelStats
    );

    std::cout << '\n';

    printStats(
        "End-to-end query latency",
        benchmark.totalStats
    );

    std::cout
        << "\nTransfer P50\n"
        << "  Query upload:    "
        << benchmark.uploadStats.p50
        << " ms\n"
        << "  Result download: "
        << benchmark.downloadStats.p50
        << " ms\n";
}

// =========================================================
// Main
// =========================================================

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

            return EXIT_FAILURE;
        }

        // -------------------------------------------------
        // CUDA device
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

        std::cout
            << "====================================\n"
            << "      Vortex CUDA Benchmark\n"
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
            << " x "
            << dimension
            << '\n';

        std::cout
            << "Top-K: "
            << topK
            << '\n';

        std::cout
            << "Warm-up runs: "
            << WARMUP_RUNS
            << '\n';

        std::cout
            << "Benchmark runs: "
            << BENCHMARK_RUNS
            << "\n\n";

        // -------------------------------------------------
        // Dataset
        // -------------------------------------------------

        const std::size_t databaseElements =
            static_cast<std::size_t>(
                vectorCount
            ) *
            static_cast<std::size_t>(
                dimension
            );

        std::cout
            << "Generating deterministic dataset...\n";

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
        // CPU reference
        // -------------------------------------------------

        std::cout
            << "Running CPU reference...\n";

        const auto cpuStart =
            std::chrono::steady_clock::now();

        const std::vector<float> cpuDistances =
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
        // Persistent index
        // -------------------------------------------------

        std::cout
            << "Building persistent GPU index...\n";

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
            << "Index size:       "
            << indexSizeMiB
            << " MiB\n";

        std::cout
            << "One-time upload:  "
            << gpuIndex.indexUploadTimeMs()
            << " ms\n";

        // -------------------------------------------------
        // Shared-memory benchmark
        // -------------------------------------------------

        std::cout
            << "\nBenchmarking shared-memory kernel...\n";

        const KernelBenchmark sharedBenchmark =
            benchmarkKernel(
                gpuIndex,
                query,
                DistanceKernel::SharedMemory
            );

        // -------------------------------------------------
        // Warp-shuffle benchmark
        // -------------------------------------------------

        std::cout
            << "Benchmarking warp-shuffle kernel...\n";

        const KernelBenchmark warpBenchmark =
            benchmarkKernel(
                gpuIndex,
                query,
                DistanceKernel::WarpShuffle
            );

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        const float sharedError =
            calculateMaxAbsoluteError(
                cpuDistances,
                sharedBenchmark.finalDistances
            );

        const float warpError =
            calculateMaxAbsoluteError(
                cpuDistances,
                warpBenchmark.finalDistances
            );

        const float kernelDifference =
            calculateMaxAbsoluteError(
                sharedBenchmark.finalDistances,
                warpBenchmark.finalDistances
            );

        const std::vector<int> cpuTopK =
            findTopK(
                cpuDistances,
                topK
            );

        const std::vector<int> sharedTopK =
            findTopK(
                sharedBenchmark.finalDistances,
                topK
            );

        const std::vector<int> warpTopK =
            findTopK(
                warpBenchmark.finalDistances,
                topK
            );

        const bool sharedTopKMatches =
            cpuTopK ==
            sharedTopK;

        const bool warpTopKMatches =
            cpuTopK ==
            warpTopK;

        // -------------------------------------------------
        // Results
        // -------------------------------------------------

        std::cout
            << "\n\n====================================\n"
            << "           Benchmark Results\n"
            << "====================================\n";

        std::cout
            << "\nCPU reference\n"
            << "------------------------------------\n"
            << "Execution time: "
            << cpuTimeMs
            << " ms\n";

        printKernelBenchmark(
            sharedBenchmark
        );

        printKernelBenchmark(
            warpBenchmark
        );

        // -------------------------------------------------
        // A/B improvement
        // -------------------------------------------------

        std::cout
            << "\nA/B Improvement\n"
            << "------------------------------------\n";

        const double kernelP50Speedup =
            sharedBenchmark.kernelStats.p50 /
            warpBenchmark.kernelStats.p50;

        const double kernelP95Speedup =
            sharedBenchmark.kernelStats.p95 /
            warpBenchmark.kernelStats.p95;

        const double queryP50Speedup =
            sharedBenchmark.totalStats.p50 /
            warpBenchmark.totalStats.p50;

        const double cpuVsWarpP50 =
            cpuTimeMs /
            warpBenchmark.totalStats.p50;

        std::cout
            << "Warp vs Shared kernel P50: "
            << kernelP50Speedup
            << "x\n";

        std::cout
            << "Warp vs Shared kernel P95: "
            << kernelP95Speedup
            << "x\n";

        std::cout
            << "Warp vs Shared query P50:  "
            << queryP50Speedup
            << "x\n";

        std::cout
            << "CPU vs Warp query P50:     "
            << cpuVsWarpP50
            << "x\n";

        // -------------------------------------------------
        // Correctness output
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "------------------------------------\n";

        std::cout
            << std::setprecision(8);

        std::cout
            << "Shared max error:     "
            << sharedError
            << '\n';

        std::cout
            << "Warp max error:       "
            << warpError
            << '\n';

        std::cout
            << "Shared/Warp delta:    "
            << kernelDifference
            << '\n';

        std::cout
            << "Shared Top-K match:   "
            << (
                sharedTopKMatches
                    ? "YES"
                    : "NO"
            )
            << '\n';

        std::cout
            << "Warp Top-K match:     "
            << (
                warpTopKMatches
                    ? "YES"
                    : "NO"
            )
            << '\n';

        if (!sharedTopKMatches ||
            !warpTopKMatches)
        {
            std::cerr
                << "\nCorrectness validation failed.\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\nMilestone 3 completed successfully.\n";

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