#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "gpu_vector_index.cuh"

#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

namespace
{

constexpr int DEFAULT_VECTOR_COUNT =
    100000;

constexpr int DEFAULT_DIMENSION =
    128;

constexpr int DEFAULT_TOP_K =
    10;

constexpr int WARMUP_RUNS =
    5;

constexpr int BENCHMARK_RUNS =
    50;

constexpr float MAX_ALLOWED_ERROR =
    0.001F;

constexpr std::array<int, 4>
    WARP_BLOCK_SIZES{
        32,
        64,
        128,
        256
    };

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
    GpuSearchConfig config{};

    TimingStats kernelStats;
    TimingStats totalStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<float> finalDistances;
};

struct CorrectnessResult
{
    float maxError = 0.0F;
    bool numericPass = false;
    bool topKMatch = false;
};

} // namespace

// =========================================================
// Dataset generation
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
// CPU exact L2 reference
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
        float distance =
            0.0F;

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
        ] =
            distance;
    }

    return distances;
}

// =========================================================
// Numerical correctness
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

    float maxError =
        0.0F;

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
// CPU Top-K reference
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
                return
                    left <
                    right;
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
            "from empty values."
        );
    }

    if (sortedValues.size() == 1)
    {
        return
            sortedValues.front();
    }

    const double position =
        percentileValue *
        static_cast<double>(
            sortedValues.size() -
            1
        );

    const std::size_t lowerIndex =
        static_cast<std::size_t>(
            std::floor(
                position
            )
        );

    const std::size_t upperIndex =
        static_cast<std::size_t>(
            std::ceil(
                position
            )
        );

    if (lowerIndex ==
        upperIndex)
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
        sortedValues[
            lowerIndex
        ] +
        fraction *
        (
            sortedValues[
                upperIndex
            ] -
            sortedValues[
                lowerIndex
            ]
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
// Kernel names
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
// Benchmark one launch configuration
// =========================================================

KernelBenchmark benchmarkKernel(
    GpuVectorIndex& index,
    const std::vector<float>& query,
    const GpuSearchConfig& config)
{
    index.warmUp(
        config,
        WARMUP_RUNS
    );

    std::vector<double>
        kernelTimes;

    std::vector<double>
        totalTimes;

    std::vector<double>
        uploadTimes;

    std::vector<double>
        downloadTimes;

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

    std::vector<float>
        finalDistances;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuSearchResult result =
            index.search(
                query,
                config
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

    benchmark.config =
        config;

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
// Correctness evaluation
// =========================================================

CorrectnessResult evaluateCorrectness(
    const std::vector<float>& cpuDistances,
    const std::vector<int>& cpuTopK,
    const std::vector<float>& gpuDistances,
    const int topK)
{
    CorrectnessResult result;

    result.maxError =
        calculateMaxAbsoluteError(
            cpuDistances,
            gpuDistances
        );

    result.numericPass =
        std::isfinite(
            result.maxError
        ) &&
        result.maxError <=
            MAX_ALLOWED_ERROR;

    const std::vector<int>
        gpuTopK =
            findTopK(
                gpuDistances,
                topK
            );

    result.topKMatch =
        cpuTopK ==
        gpuTopK;

    return result;
}

// =========================================================
// Print timing statistics
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

// =========================================================
// Print one kernel benchmark
// =========================================================

void printKernelBenchmark(
    const KernelBenchmark& benchmark)
{
    std::cout
        << "\n"
        << kernelName(
            benchmark.config.kernel
        )
        << " / "
        << benchmark.config.blockSize
        << " threads\n";

    std::cout
        << "------------------------------------\n";

    printStats(
        "Kernel latency",
        benchmark.kernelStats
    );

    std::cout
        << '\n';

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

        // -------------------------------------------------
        // CLI
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
                << "Invalid arguments.\n"
                << "Usage: vortex_cuda "
                << "[vectorCount] "
                << "[dimension] "
                << "[topK]\n";

            return EXIT_FAILURE;
        }

        // -------------------------------------------------
        // CUDA device discovery
        // -------------------------------------------------

        int deviceCount =
            0;

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
            << "============================================\n"
            << "       Vortex CUDA Launch Autotuner\n"
            << "============================================\n\n";

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
            << "Warm-up runs/config: "
            << WARMUP_RUNS
            << '\n';

        std::cout
            << "Benchmark runs/config: "
            << BENCHMARK_RUNS
            << "\n\n";

        // -------------------------------------------------
        // Dataset
        // -------------------------------------------------

        const std::size_t
            databaseElements =
                static_cast<std::size_t>(
                    vectorCount
                ) *
                static_cast<std::size_t>(
                    dimension
                );

        std::cout
            << "Generating deterministic dataset...\n";

        const std::vector<float>
            database =
                generateRandomData(
                    databaseElements,
                    42
                );

        const std::vector<float>
            query =
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

        const std::vector<int>
            cpuTopK =
                findTopK(
                    cpuDistances,
                    topK
                );

        // -------------------------------------------------
        // Persistent GPU index
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
        // Historical shared-memory baseline
        // -------------------------------------------------

        const GpuSearchConfig
            sharedConfig{
                DistanceKernel::SharedMemory,
                256
            };

        std::cout
            << "\nBenchmarking shared-memory "
            << "256-thread baseline...\n";

        const KernelBenchmark
            sharedBenchmark =
                benchmarkKernel(
                    gpuIndex,
                    query,
                    sharedConfig
                );

        // -------------------------------------------------
        // Warp block-size autotuning
        // -------------------------------------------------

        std::vector<KernelBenchmark>
            warpBenchmarks;

        warpBenchmarks.reserve(
            WARP_BLOCK_SIZES.size()
        );

        for (const int blockSize :
             WARP_BLOCK_SIZES)
        {
            std::cout
                << "Benchmarking warp-shuffle "
                << blockSize
                << "-thread configuration...\n";

            const GpuSearchConfig
                config{
                    DistanceKernel::WarpShuffle,
                    blockSize
                };

            warpBenchmarks.push_back(
                benchmarkKernel(
                    gpuIndex,
                    query,
                    config
                )
            );
        }

        // -------------------------------------------------
        // Select lowest kernel P50
        // -------------------------------------------------

        const auto bestIterator =
            std::min_element(
                warpBenchmarks.begin(),
                warpBenchmarks.end(),
                [](
                    const KernelBenchmark& left,
                    const KernelBenchmark& right)
                {
                    return
                        left.kernelStats.p50 <
                        right.kernelStats.p50;
                }
            );

        if (bestIterator ==
            warpBenchmarks.end())
        {
            throw std::runtime_error(
                "No warp benchmark result available."
            );
        }

        const KernelBenchmark&
            bestWarpBenchmark =
                *bestIterator;

        const auto warp256Iterator =
            std::find_if(
                warpBenchmarks.begin(),
                warpBenchmarks.end(),
                [](
                    const KernelBenchmark& benchmark)
                {
                    return
                        benchmark.config.blockSize ==
                        256;
                }
            );

        if (warp256Iterator ==
            warpBenchmarks.end())
        {
            throw std::runtime_error(
                "256-thread warp baseline missing."
            );
        }

        const KernelBenchmark&
            warp256Benchmark =
                *warp256Iterator;

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        const CorrectnessResult
            sharedCorrectness =
                evaluateCorrectness(
                    cpuDistances,
                    cpuTopK,
                    sharedBenchmark.finalDistances,
                    topK
                );

        std::vector<CorrectnessResult>
            warpCorrectness;

        warpCorrectness.reserve(
            warpBenchmarks.size()
        );

        bool allCorrect =
            sharedCorrectness.numericPass &&
            sharedCorrectness.topKMatch;

        for (const KernelBenchmark&
             benchmark :
             warpBenchmarks)
        {
            const CorrectnessResult
                correctness =
                    evaluateCorrectness(
                        cpuDistances,
                        cpuTopK,
                        benchmark.finalDistances,
                        topK
                    );

            warpCorrectness.push_back(
                correctness
            );

            allCorrect =
                allCorrect &&
                correctness.numericPass &&
                correctness.topKMatch;
        }

        // -------------------------------------------------
        // Benchmark output
        // -------------------------------------------------

        std::cout
            << "\n\n============================================\n"
            << "              Benchmark Results\n"
            << "============================================\n";

        std::cout
            << "\nCPU reference\n"
            << "------------------------------------\n"
            << "Execution time: "
            << cpuTimeMs
            << " ms\n";

        printKernelBenchmark(
            sharedBenchmark
        );

        for (const KernelBenchmark&
             benchmark :
             warpBenchmarks)
        {
            printKernelBenchmark(
                benchmark
            );
        }

        // -------------------------------------------------
        // Block-size comparison table
        // -------------------------------------------------

        std::cout
            << "\nWarp Block-Size Comparison\n"
            << "--------------------------------------------\n"
            << "Threads     Kernel P50     P95       Query P50\n";

        for (const KernelBenchmark&
             benchmark :
             warpBenchmarks)
        {
            std::cout
                << std::setw(7)
                << benchmark.config.blockSize
                << "     "
                << std::setw(10)
                << benchmark.kernelStats.p50
                << "     "
                << std::setw(7)
                << benchmark.kernelStats.p95
                << "     "
                << std::setw(9)
                << benchmark.totalStats.p50
                << '\n';
        }

        // -------------------------------------------------
        // Autotuner result
        // -------------------------------------------------

        const double
            vsWarp256KernelSpeedup =
                warp256Benchmark.kernelStats.p50 /
                bestWarpBenchmark.kernelStats.p50;

        const double
            vsWarp256QuerySpeedup =
                warp256Benchmark.totalStats.p50 /
                bestWarpBenchmark.totalStats.p50;

        const double
            vsSharedKernelSpeedup =
                sharedBenchmark.kernelStats.p50 /
                bestWarpBenchmark.kernelStats.p50;

        const double
            cpuVsBestQuerySpeedup =
                cpuTimeMs /
                bestWarpBenchmark.totalStats.p50;

        std::cout
            << "\nAutotuned Selection\n"
            << "--------------------------------------------\n";

        std::cout
            << "Vector dimension:          "
            << dimension
            << '\n';

        std::cout
            << "Selected block size:       "
            << bestWarpBenchmark.config.blockSize
            << " threads\n";

        std::cout
            << "Selection metric:          "
            << "lowest kernel P50\n";

        std::cout
            << "Selected kernel P50:       "
            << bestWarpBenchmark.kernelStats.p50
            << " ms\n";

        std::cout
            << "Selected kernel P95:       "
            << bestWarpBenchmark.kernelStats.p95
            << " ms\n";

        std::cout
            << "Selected query P50:        "
            << bestWarpBenchmark.totalStats.p50
            << " ms\n";

        std::cout
            << "vs Warp-256 kernel P50:    "
            << vsWarp256KernelSpeedup
            << "x\n";

        std::cout
            << "vs Warp-256 query P50:     "
            << vsWarp256QuerySpeedup
            << "x\n";

        std::cout
            << "vs Shared-256 kernel P50:  "
            << vsSharedKernelSpeedup
            << "x\n";

        std::cout
            << "CPU vs selected query P50: "
            << cpuVsBestQuerySpeedup
            << "x\n";

        // -------------------------------------------------
        // Correctness table
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "--------------------------------------------\n";

        std::cout
            << std::setprecision(8);

        std::cout
            << "Shared-256"
            << "  error="
            << sharedCorrectness.maxError
            << "  numeric="
            << (
                sharedCorrectness.numericPass
                    ? "PASS"
                    : "FAIL"
            )
            << "  topK="
            << (
                sharedCorrectness.topKMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        for (std::size_t index = 0;
             index < warpBenchmarks.size();
             ++index)
        {
            std::cout
                << "Warp-"
                << std::setw(3)
                << warpBenchmarks[
                    index
                ].config.blockSize
                << "    error="
                << warpCorrectness[
                    index
                ].maxError
                << "  numeric="
                << (
                    warpCorrectness[
                        index
                    ].numericPass
                        ? "PASS"
                        : "FAIL"
                )
                << "  topK="
                << (
                    warpCorrectness[
                        index
                    ].topKMatch
                        ? "PASS"
                        : "FAIL"
                )
                << '\n';
        }

        if (!allCorrect)
        {
            std::cerr
                << "\nCorrectness validation failed.\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\nMilestone 4 completed successfully.\n";

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