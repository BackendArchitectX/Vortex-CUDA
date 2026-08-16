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

constexpr int AUTOTUNE_WARMUP_RUNS =
    2;

constexpr int AUTOTUNE_RUNS =
    10;

constexpr int CPU_WARMUP_RUNS =
    2;

constexpr int GPU_WARMUP_RUNS =
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

struct CpuBenchmark
{
    TimingStats totalStats;
    std::vector<Neighbor> finalNeighbors;
};

struct GpuBenchmark
{
    TimingStats distanceStats;
    TimingStats topKStats;
    TimingStats totalStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<Neighbor> finalNeighbors;

    int topKPasses = 0;
};

struct AutotuneEntry
{
    int blockSize = 0;
    TimingStats kernelStats;
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
// CPU distance calculation
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

        for (int dimensionIndex = 0;
             dimensionIndex < dimension;
             ++dimensionIndex)
        {
            const float difference =
                database[
                    offset +
                    static_cast<std::size_t>(
                        dimensionIndex
                    )
                ] -
                query[
                    static_cast<std::size_t>(
                        dimensionIndex
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
// CPU Top-K
// =========================================================

std::vector<Neighbor> findTopKNeighbors(
    const std::vector<float>& distances,
    const int topK)
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
        indices.begin() + topK,
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
                return
                    left <
                    right;
            }

            return
                leftDistance <
                rightDistance;
        }
    );

    std::vector<Neighbor> neighbors;

    neighbors.reserve(
        static_cast<std::size_t>(
            topK
        )
    );

    for (int rank = 0;
         rank < topK;
         ++rank)
    {
        const int index =
            indices[
                static_cast<std::size_t>(
                    rank
                )
            ];

        neighbors.push_back(
            Neighbor{
                index,
                distances[
                    static_cast<std::size_t>(
                        index
                    )
                ]
            }
        );
    }

    return neighbors;
}

// =========================================================
// Complete CPU exact search
// =========================================================

std::vector<Neighbor> searchCpuExact(
    const std::vector<float>& database,
    const std::vector<float>& query,
    const int vectorCount,
    const int dimension,
    const int topK)
{
    const std::vector<float> distances =
        calculateDistancesCpu(
            database,
            query,
            vectorCount,
            dimension
        );

    return
        findTopKNeighbors(
            distances,
            topK
        );
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
            "from empty data."
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
            "from empty data."
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
// Autotune distance block size
// =========================================================

GpuSearchConfig autotuneBlockSize(
    GpuVectorIndex& index,
    const std::vector<float>& query,
    std::vector<AutotuneEntry>& entries)
{
    entries.clear();

    double bestP50 =
        std::numeric_limits<
            double
        >::infinity();

    int bestBlockSize =
        256;

    for (const int blockSize :
         WARP_BLOCK_SIZES)
    {
        const GpuSearchConfig config{
            DistanceKernel::WarpShuffle,
            blockSize
        };

        index.warmUp(
            config,
            AUTOTUNE_WARMUP_RUNS
        );

        std::vector<double> times;

        times.reserve(
            AUTOTUNE_RUNS
        );

        for (int run = 0;
             run < AUTOTUNE_RUNS;
             ++run)
        {
            const GpuSearchResult result =
                index.search(
                    query,
                    config
                );

            times.push_back(
                static_cast<double>(
                    result.kernelTimeMs
                )
            );
        }

        const TimingStats stats =
            calculateStats(
                times
            );

        entries.push_back(
            AutotuneEntry{
                blockSize,
                stats
            }
        );

        if (stats.p50 <
            bestP50)
        {
            bestP50 =
                stats.p50;

            bestBlockSize =
                blockSize;
        }
    }

    return
        GpuSearchConfig{
            DistanceKernel::WarpShuffle,
            bestBlockSize
        };
}

// =========================================================
// CPU benchmark
// =========================================================

CpuBenchmark benchmarkCpu(
    const std::vector<float>& database,
    const std::vector<float>& query,
    const int vectorCount,
    const int dimension,
    const int topK)
{
    for (int run = 0;
         run < CPU_WARMUP_RUNS;
         ++run)
    {
        (void)searchCpuExact(
            database,
            query,
            vectorCount,
            dimension,
            topK
        );
    }

    std::vector<double> times;

    times.reserve(
        BENCHMARK_RUNS
    );

    std::vector<Neighbor>
        finalNeighbors;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        const auto start =
            std::chrono::steady_clock::now();

        std::vector<Neighbor> neighbors =
            searchCpuExact(
                database,
                query,
                vectorCount,
                dimension,
                topK
            );

        const auto end =
            std::chrono::steady_clock::now();

        times.push_back(
            std::chrono::duration<
                double,
                std::milli
            >(
                end -
                start
            ).count()
        );

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            finalNeighbors =
                std::move(
                    neighbors
                );
        }
    }

    CpuBenchmark benchmark;

    benchmark.totalStats =
        calculateStats(
            times
        );

    benchmark.finalNeighbors =
        std::move(
            finalNeighbors
        );

    return benchmark;
}

// =========================================================
// GPU Top-K benchmark
// =========================================================

GpuBenchmark benchmarkGpu(
    GpuVectorIndex& index,
    const std::vector<float>& query,
    const int topK,
    const GpuSearchConfig& config)
{
    index.warmUpTopK(
        config,
        topK,
        GPU_WARMUP_RUNS
    );

    std::vector<double>
        distanceTimes;

    std::vector<double>
        topKTimes;

    std::vector<double>
        totalTimes;

    std::vector<double>
        uploadTimes;

    std::vector<double>
        downloadTimes;

    distanceTimes.reserve(
        BENCHMARK_RUNS
    );

    topKTimes.reserve(
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

    std::vector<Neighbor>
        finalNeighbors;

    int topKPasses =
        0;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuTopKResult result =
            index.searchTopK(
                query,
                topK,
                config
            );

        distanceTimes.push_back(
            static_cast<double>(
                result.distanceKernelTimeMs
            )
        );

        topKTimes.push_back(
            static_cast<double>(
                result.topKKernelTimeMs
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

        topKPasses =
            result.topKPasses;

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            finalNeighbors =
                std::move(
                    result.neighbors
                );
        }
    }

    GpuBenchmark benchmark;

    benchmark.distanceStats =
        calculateStats(
            distanceTimes
        );

    benchmark.topKStats =
        calculateStats(
            topKTimes
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

    benchmark.finalNeighbors =
        std::move(
            finalNeighbors
        );

    benchmark.topKPasses =
        topKPasses;

    return benchmark;
}

// =========================================================
// Correctness
// =========================================================

bool neighborIndicesMatch(
    const std::vector<Neighbor>& expected,
    const std::vector<Neighbor>& actual)
{
    if (expected.size() !=
        actual.size())
    {
        return false;
    }

    for (std::size_t index = 0;
         index < expected.size();
         ++index)
    {
        if (expected[index].index !=
            actual[index].index)
        {
            return false;
        }
    }

    return true;
}

float maxNeighborDistanceError(
    const std::vector<Neighbor>& expected,
    const std::vector<Neighbor>& actual)
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
                expected[index].distance -
                actual[index].distance
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
// Printing
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

void printNeighbors(
    const std::vector<Neighbor>& neighbors)
{
    std::cout
        << "\nTop-K nearest vectors\n"
        << "--------------------------------------------\n";

    for (std::size_t rank = 0;
         rank < neighbors.size();
         ++rank)
    {
        std::cout
            << "#"
            << std::setw(2)
            << rank + 1
            << "  vector="
            << std::setw(7)
            << neighbors[
                rank
            ].index
            << "  squared_l2="
            << std::fixed
            << std::setprecision(6)
            << neighbors[
                rank
            ].distance
            << '\n';
    }
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
            topK > vectorCount ||
            topK >
                GpuVectorIndex::MAX_TOP_K)
        {
            std::cerr
                << "Invalid arguments.\n"
                << "Usage: vortex_cuda "
                << "[vectorCount] "
                << "[dimension] "
                << "[topK]\n"
                << "Current GPU Top-K implementation "
                << "supports K <= "
                << GpuVectorIndex::MAX_TOP_K
                << ".\n";

            return EXIT_FAILURE;
        }

        // -------------------------------------------------
        // CUDA device
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

        std::cout
            << "============================================\n"
            << "       Vortex CUDA Exact GPU Top-K\n"
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
        // GPU index
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
        // Autotune distance kernel
        // -------------------------------------------------

        std::cout
            << "\nAutotuning distance kernel...\n";

        std::vector<AutotuneEntry>
            autotuneEntries;

        const GpuSearchConfig selectedConfig =
            autotuneBlockSize(
                gpuIndex,
                query,
                autotuneEntries
            );

        std::cout
            << "\nAutotune Results\n"
            << "--------------------------------------------\n"
            << "Threads     Kernel P50     P95\n";

        for (const AutotuneEntry&
             entry :
             autotuneEntries)
        {
            std::cout
                << std::setw(7)
                << entry.blockSize
                << "     "
                << std::setw(10)
                << entry.kernelStats.p50
                << "     "
                << std::setw(7)
                << entry.kernelStats.p95
                << '\n';
        }

        std::cout
            << "\nSelected block size: "
            << selectedConfig.blockSize
            << " threads\n";

        // -------------------------------------------------
        // Independent correctness references
        // -------------------------------------------------

        std::cout
            << "\nBuilding correctness references...\n";

        const std::vector<Neighbor>
            cpuReference =
                searchCpuExact(
                    database,
                    query,
                    vectorCount,
                    dimension,
                    topK
                );

        const GpuSearchResult
            fullGpuResult =
                gpuIndex.search(
                    query,
                    selectedConfig
                );

        const std::vector<Neighbor>
            fullGpuReference =
                findTopKNeighbors(
                    fullGpuResult.distances,
                    topK
                );

        // -------------------------------------------------
        // CPU complete search benchmark
        // -------------------------------------------------

        std::cout
            << "Benchmarking complete CPU exact search...\n";

        const CpuBenchmark cpuBenchmark =
            benchmarkCpu(
                database,
                query,
                vectorCount,
                dimension,
                topK
            );

        // -------------------------------------------------
        // GPU complete search benchmark
        // -------------------------------------------------

        std::cout
            << "Benchmarking hierarchical GPU Top-K search...\n";

        const GpuBenchmark gpuBenchmark =
            benchmarkGpu(
                gpuIndex,
                query,
                topK,
                selectedConfig
            );

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        const bool cpuGpuIndicesMatch =
            neighborIndicesMatch(
                cpuReference,
                gpuBenchmark.finalNeighbors
            );

        const bool fullGpuIndicesMatch =
            neighborIndicesMatch(
                fullGpuReference,
                gpuBenchmark.finalNeighbors
            );

        const float maxDistanceError =
            maxNeighborDistanceError(
                cpuReference,
                gpuBenchmark.finalNeighbors
            );

        const bool numericPass =
            std::isfinite(
                maxDistanceError
            ) &&
            maxDistanceError <=
                MAX_ALLOWED_ERROR;

        // -------------------------------------------------
        // Data transfer reduction
        // -------------------------------------------------

        const std::size_t oldDownloadBytes =
            static_cast<std::size_t>(
                vectorCount
            ) *
            sizeof(float);

        const std::size_t newDownloadBytes =
            static_cast<std::size_t>(
                topK
            ) *
            sizeof(Neighbor);

        const double transferReduction =
            static_cast<double>(
                oldDownloadBytes
            ) /
            static_cast<double>(
                newDownloadBytes
            );

        // -------------------------------------------------
        // Results
        // -------------------------------------------------

        std::cout
            << "\n\n============================================\n"
            << "              Benchmark Results\n"
            << "============================================\n";

        std::cout
            << "\nCPU Complete Exact Search\n"
            << "--------------------------------------------\n";

        printStats(
            "Distance + CPU Top-K",
            cpuBenchmark.totalStats
        );

        std::cout
            << "\nGPU Complete Exact Search\n"
            << "--------------------------------------------\n";

        printStats(
            "Distance kernel",
            gpuBenchmark.distanceStats
        );

        std::cout
            << '\n';

        printStats(
            "Hierarchical Top-K",
            gpuBenchmark.topKStats
        );

        std::cout
            << '\n';

        printStats(
            "End-to-end exact search",
            gpuBenchmark.totalStats
        );

        std::cout
            << "\nTransfer P50\n"
            << "  Query upload:    "
            << gpuBenchmark.uploadStats.p50
            << " ms\n"
            << "  Top-K download:  "
            << gpuBenchmark.downloadStats.p50
            << " ms\n";

        // -------------------------------------------------
        // Search speedup
        // -------------------------------------------------

        const double searchP50Speedup =
            cpuBenchmark.totalStats.p50 /
            gpuBenchmark.totalStats.p50;

        const double searchP95Speedup =
            cpuBenchmark.totalStats.p95 /
            gpuBenchmark.totalStats.p95;

        std::cout
            << "\nExact Search Speedup\n"
            << "--------------------------------------------\n";

        std::cout
            << "CPU P50:                 "
            << cpuBenchmark.totalStats.p50
            << " ms\n";

        std::cout
            << "GPU P50:                 "
            << gpuBenchmark.totalStats.p50
            << " ms\n";

        std::cout
            << "CPU vs GPU P50:          "
            << searchP50Speedup
            << "x\n";

        std::cout
            << "CPU vs GPU P95:          "
            << searchP95Speedup
            << "x\n";

        // -------------------------------------------------
        // Top-K hierarchy
        // -------------------------------------------------

        std::cout
            << "\nGPU Top-K Architecture\n"
            << "--------------------------------------------\n";

        std::cout
            << "Hierarchy passes:        "
            << gpuBenchmark.topKPasses
            << '\n';

        std::cout
            << "Old result download:     "
            << oldDownloadBytes
            << " bytes\n";

        std::cout
            << "New Top-K download:      "
            << newDownloadBytes
            << " bytes\n";

        std::cout
            << "Transfer reduction:      "
            << transferReduction
            << "x\n";

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "--------------------------------------------\n";

        std::cout
            << std::setprecision(8);

        std::cout
            << "CPU vs GPU Top-K indices: "
            << (
                cpuGpuIndicesMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Full GPU vs Top-K path:   "
            << (
                fullGpuIndicesMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Max distance error:       "
            << maxDistanceError
            << '\n';

        std::cout
            << "Numeric validation:       "
            << (
                numericPass
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        printNeighbors(
            gpuBenchmark.finalNeighbors
        );

        if (!cpuGpuIndicesMatch ||
            !fullGpuIndicesMatch ||
            !numericPass)
        {
            std::cerr
                << "\nCorrectness validation failed.\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\nMilestone 5 completed successfully.\n";

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