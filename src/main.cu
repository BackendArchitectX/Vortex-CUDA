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

struct BaselineGpuBenchmark
{
    TimingStats distanceStats;
    TimingStats topKStats;
    TimingStats totalStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<Neighbor> finalNeighbors;

    int topKPasses = 0;
};

struct FusedGpuBenchmark
{
    TimingStats fusedStats;
    TimingStats mergeStats;
    TimingStats totalStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<Neighbor> finalNeighbors;

    int topKPasses = 0;
};

struct GpuComparisonBenchmark
{
    BaselineGpuBenchmark baseline;
    FusedGpuBenchmark fused;
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
// Distance kernel autotuning
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
// Interleaved GPU A/B benchmark
// =========================================================

GpuComparisonBenchmark benchmarkGpuComparison(
    GpuVectorIndex& index,
    const std::vector<float>& query,
    const int topK,
    const GpuSearchConfig& baselineConfig)
{
    index.warmUpTopK(
        baselineConfig,
        topK,
        GPU_WARMUP_RUNS
    );

    index.warmUpFusedTopK(
        topK,
        GPU_WARMUP_RUNS
    );

    std::vector<double>
        baselineDistanceTimes;

    std::vector<double>
        baselineTopKTimes;

    std::vector<double>
        baselineTotalTimes;

    std::vector<double>
        baselineUploadTimes;

    std::vector<double>
        baselineDownloadTimes;

    std::vector<double>
        fusedKernelTimes;

    std::vector<double>
        fusedMergeTimes;

    std::vector<double>
        fusedTotalTimes;

    std::vector<double>
        fusedUploadTimes;

    std::vector<double>
        fusedDownloadTimes;

    baselineDistanceTimes.reserve(
        BENCHMARK_RUNS
    );

    baselineTopKTimes.reserve(
        BENCHMARK_RUNS
    );

    baselineTotalTimes.reserve(
        BENCHMARK_RUNS
    );

    baselineUploadTimes.reserve(
        BENCHMARK_RUNS
    );

    baselineDownloadTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedKernelTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedMergeTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedTotalTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedUploadTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedDownloadTimes.reserve(
        BENCHMARK_RUNS
    );

    std::vector<Neighbor>
        baselineFinalNeighbors;

    std::vector<Neighbor>
        fusedFinalNeighbors;

    int baselinePasses =
        0;

    int fusedPasses =
        0;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuTopKResult baselineResult;
        GpuFusedTopKResult fusedResult;

        // Alternate execution order to reduce
        // systematic temperature/order bias.
        if ((run & 1) == 0)
        {
            baselineResult =
                index.searchTopK(
                    query,
                    topK,
                    baselineConfig
                );

            fusedResult =
                index.searchTopKFused(
                    query,
                    topK
                );
        }
        else
        {
            fusedResult =
                index.searchTopKFused(
                    query,
                    topK
                );

            baselineResult =
                index.searchTopK(
                    query,
                    topK,
                    baselineConfig
                );
        }

        baselineDistanceTimes.push_back(
            static_cast<double>(
                baselineResult.distanceKernelTimeMs
            )
        );

        baselineTopKTimes.push_back(
            static_cast<double>(
                baselineResult.topKKernelTimeMs
            )
        );

        baselineTotalTimes.push_back(
            baselineResult.totalTimeMs
        );

        baselineUploadTimes.push_back(
            baselineResult.queryUploadTimeMs
        );

        baselineDownloadTimes.push_back(
            baselineResult.resultDownloadTimeMs
        );

        fusedKernelTimes.push_back(
            static_cast<double>(
                fusedResult.fusedKernelTimeMs
            )
        );

        fusedMergeTimes.push_back(
            static_cast<double>(
                fusedResult.mergeKernelTimeMs
            )
        );

        fusedTotalTimes.push_back(
            fusedResult.totalTimeMs
        );

        fusedUploadTimes.push_back(
            fusedResult.queryUploadTimeMs
        );

        fusedDownloadTimes.push_back(
            fusedResult.resultDownloadTimeMs
        );

        baselinePasses =
            baselineResult.topKPasses;

        fusedPasses =
            fusedResult.topKPasses;

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            baselineFinalNeighbors =
                std::move(
                    baselineResult.neighbors
                );

            fusedFinalNeighbors =
                std::move(
                    fusedResult.neighbors
                );
        }
    }

    GpuComparisonBenchmark benchmark;

    benchmark.baseline.distanceStats =
        calculateStats(
            baselineDistanceTimes
        );

    benchmark.baseline.topKStats =
        calculateStats(
            baselineTopKTimes
        );

    benchmark.baseline.totalStats =
        calculateStats(
            baselineTotalTimes
        );

    benchmark.baseline.uploadStats =
        calculateStats(
            baselineUploadTimes
        );

    benchmark.baseline.downloadStats =
        calculateStats(
            baselineDownloadTimes
        );

    benchmark.baseline.finalNeighbors =
        std::move(
            baselineFinalNeighbors
        );

    benchmark.baseline.topKPasses =
        baselinePasses;

    benchmark.fused.fusedStats =
        calculateStats(
            fusedKernelTimes
        );

    benchmark.fused.mergeStats =
        calculateStats(
            fusedMergeTimes
        );

    benchmark.fused.totalStats =
        calculateStats(
            fusedTotalTimes
        );

    benchmark.fused.uploadStats =
        calculateStats(
            fusedUploadTimes
        );

    benchmark.fused.downloadStats =
        calculateStats(
            fusedDownloadTimes
        );

    benchmark.fused.finalNeighbors =
        std::move(
            fusedFinalNeighbors
        );

    benchmark.fused.topKPasses =
        fusedPasses;

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
                << "GPU Top-K supports K <= "
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
            << "     Vortex CUDA Fused Exact GPU Top-K\n"
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
            << "Fused candidates/block: "
            << GpuVectorIndex::
                FUSED_CANDIDATES_PER_BLOCK
            << '\n';

        std::cout
            << "Benchmark runs/path: "
            << BENCHMARK_RUNS
            << '\n';

        std::cout
            << "A/B ordering: alternating\n\n";

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
        // Baseline distance autotuning
        // -------------------------------------------------

        std::cout
            << "\nAutotuning Milestone 5 "
            << "distance kernel...\n";

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
            << "\nBaseline selected block size: "
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
        // CPU benchmark
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
        // GPU A/B benchmark
        // -------------------------------------------------

        std::cout
            << "Benchmarking Milestone 5 baseline vs "
            << "Milestone 6 fused path...\n";

        const GpuComparisonBenchmark
            gpuBenchmark =
                benchmarkGpuComparison(
                    gpuIndex,
                    query,
                    topK,
                    selectedConfig
                );

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        const bool cpuBaselineMatch =
            neighborIndicesMatch(
                cpuReference,
                gpuBenchmark
                    .baseline
                    .finalNeighbors
            );

        const bool cpuFusedMatch =
            neighborIndicesMatch(
                cpuReference,
                gpuBenchmark
                    .fused
                    .finalNeighbors
            );

        const bool fullGpuBaselineMatch =
            neighborIndicesMatch(
                fullGpuReference,
                gpuBenchmark
                    .baseline
                    .finalNeighbors
            );

        const bool fullGpuFusedMatch =
            neighborIndicesMatch(
                fullGpuReference,
                gpuBenchmark
                    .fused
                    .finalNeighbors
            );

        const bool baselineFusedMatch =
            neighborIndicesMatch(
                gpuBenchmark
                    .baseline
                    .finalNeighbors,
                gpuBenchmark
                    .fused
                    .finalNeighbors
            );

        const float baselineDistanceError =
            maxNeighborDistanceError(
                cpuReference,
                gpuBenchmark
                    .baseline
                    .finalNeighbors
            );

        const float fusedDistanceError =
            maxNeighborDistanceError(
                cpuReference,
                gpuBenchmark
                    .fused
                    .finalNeighbors
            );

        const bool baselineNumericPass =
            std::isfinite(
                baselineDistanceError
            ) &&
            baselineDistanceError <=
                MAX_ALLOWED_ERROR;

        const bool fusedNumericPass =
            std::isfinite(
                fusedDistanceError
            ) &&
            fusedDistanceError <=
                MAX_ALLOWED_ERROR;

        const bool allCorrect =
            cpuBaselineMatch &&
            cpuFusedMatch &&
            fullGpuBaselineMatch &&
            fullGpuFusedMatch &&
            baselineFusedMatch &&
            baselineNumericPass &&
            fusedNumericPass;

        // -------------------------------------------------
        // Logical intermediate-memory traffic
        // -------------------------------------------------

        const std::size_t distanceArrayBytes =
            static_cast<std::size_t>(
                vectorCount
            ) *
            sizeof(float);

        const std::size_t baselineLogicalTraffic =
            distanceArrayBytes *
            2;

        const std::size_t resultDownloadBytes =
            static_cast<std::size_t>(
                topK
            ) *
            sizeof(Neighbor);

        // -------------------------------------------------
        // Results
        // -------------------------------------------------

        std::cout
            << "\n\n============================================\n"
            << "              Benchmark Results\n"
            << "============================================\n";

        // -------------------------------------------------
        // CPU
        // -------------------------------------------------

        std::cout
            << "\nCPU Complete Exact Search\n"
            << "--------------------------------------------\n";

        printStats(
            "Distance + CPU Top-K",
            cpuBenchmark.totalStats
        );

        // -------------------------------------------------
        // Milestone 5
        // -------------------------------------------------

        std::cout
            << "\nMilestone 5 GPU Baseline\n"
            << "--------------------------------------------\n";

        printStats(
            "Distance kernel",
            gpuBenchmark
                .baseline
                .distanceStats
        );

        std::cout
            << '\n';

        printStats(
            "Hierarchical Top-K",
            gpuBenchmark
                .baseline
                .topKStats
        );

        std::cout
            << '\n';

        printStats(
            "End-to-end exact search",
            gpuBenchmark
                .baseline
                .totalStats
        );

        std::cout
            << "\nTransfer P50\n"
            << "  Query upload:   "
            << gpuBenchmark
                .baseline
                .uploadStats
                .p50
            << " ms\n"
            << "  Top-K download: "
            << gpuBenchmark
                .baseline
                .downloadStats
                .p50
            << " ms\n";

        // -------------------------------------------------
        // Milestone 6
        // -------------------------------------------------

        std::cout
            << "\nMilestone 6 Fused GPU Search\n"
            << "--------------------------------------------\n";

        printStats(
            "Fused distance + local Top-K",
            gpuBenchmark
                .fused
                .fusedStats
        );

        std::cout
            << '\n';

        printStats(
            "Candidate merge",
            gpuBenchmark
                .fused
                .mergeStats
        );

        std::cout
            << '\n';

        printStats(
            "End-to-end exact search",
            gpuBenchmark
                .fused
                .totalStats
        );

        std::cout
            << "\nTransfer P50\n"
            << "  Query upload:   "
            << gpuBenchmark
                .fused
                .uploadStats
                .p50
            << " ms\n"
            << "  Top-K download: "
            << gpuBenchmark
                .fused
                .downloadStats
                .p50
            << " ms\n";

        // -------------------------------------------------
        // A/B comparison
        // -------------------------------------------------

        const double fusedVsBaselineP50 =
            gpuBenchmark
                .baseline
                .totalStats
                .p50 /
            gpuBenchmark
                .fused
                .totalStats
                .p50;

        const double fusedVsBaselineP95 =
            gpuBenchmark
                .baseline
                .totalStats
                .p95 /
            gpuBenchmark
                .fused
                .totalStats
                .p95;

        const double cpuVsBaselineP50 =
            cpuBenchmark
                .totalStats
                .p50 /
            gpuBenchmark
                .baseline
                .totalStats
                .p50;

        const double cpuVsFusedP50 =
            cpuBenchmark
                .totalStats
                .p50 /
            gpuBenchmark
                .fused
                .totalStats
                .p50;

        const double cpuVsFusedP95 =
            cpuBenchmark
                .totalStats
                .p95 /
            gpuBenchmark
                .fused
                .totalStats
                .p95;

        std::cout
            << "\nA/B Search Comparison\n"
            << "--------------------------------------------\n";

        std::cout
            << "Baseline GPU P50:        "
            << gpuBenchmark
                .baseline
                .totalStats
                .p50
            << " ms\n";

        std::cout
            << "Fused GPU P50:           "
            << gpuBenchmark
                .fused
                .totalStats
                .p50
            << " ms\n";

        std::cout
            << "Baseline GPU P95:        "
            << gpuBenchmark
                .baseline
                .totalStats
                .p95
            << " ms\n";

        std::cout
            << "Fused GPU P95:           "
            << gpuBenchmark
                .fused
                .totalStats
                .p95
            << " ms\n";

        std::cout
            << "Fused vs baseline P50:   "
            << fusedVsBaselineP50
            << "x\n";

        std::cout
            << "Fused vs baseline P95:   "
            << fusedVsBaselineP95
            << "x\n";

        std::cout
            << "CPU vs baseline P50:     "
            << cpuVsBaselineP50
            << "x\n";

        std::cout
            << "CPU vs fused P50:        "
            << cpuVsFusedP50
            << "x\n";

        std::cout
            << "CPU vs fused P95:        "
            << cpuVsFusedP95
            << "x\n";

        // -------------------------------------------------
        // Architecture
        // -------------------------------------------------

        std::cout
            << "\nMemory-Traffic Architecture\n"
            << "--------------------------------------------\n";

        std::cout
            << "Full distance array:     "
            << distanceArrayBytes
            << " bytes\n";

        std::cout
            << "Baseline logical global\n"
            << "distance write + read:   "
            << baselineLogicalTraffic
            << " bytes/query\n";

        std::cout
            << "Fused full-distance\n"
            << "materialization:         "
            << "0 bytes/query\n";

        std::cout
            << "Top-K result download:   "
            << resultDownloadBytes
            << " bytes/query\n";

        std::cout
            << "Baseline Top-K passes:   "
            << gpuBenchmark
                .baseline
                .topKPasses
            << '\n';

        std::cout
            << "Fused Top-K passes:      "
            << gpuBenchmark
                .fused
                .topKPasses
            << '\n';

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "--------------------------------------------\n";

        std::cout
            << std::setprecision(8);

        std::cout
            << "CPU vs baseline indices: "
            << (
                cpuBaselineMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "CPU vs fused indices:    "
            << (
                cpuFusedMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Full GPU vs baseline:    "
            << (
                fullGpuBaselineMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Full GPU vs fused:       "
            << (
                fullGpuFusedMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Baseline vs fused:       "
            << (
                baselineFusedMatch
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Baseline max error:      "
            << baselineDistanceError
            << '\n';

        std::cout
            << "Fused max error:         "
            << fusedDistanceError
            << '\n';

        std::cout
            << "Baseline numeric:        "
            << (
                baselineNumericPass
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        std::cout
            << "Fused numeric:           "
            << (
                fusedNumericPass
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        printNeighbors(
            gpuBenchmark
                .fused
                .finalNeighbors
        );

        if (!allCorrect)
        {
            std::cerr
                << "\nMilestone 6 correctness "
                << "validation failed.\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\nMilestone 6 validation completed successfully.\n";

        if (fusedVsBaselineP50 >
            1.0)
        {
            std::cout
                << "Fused path improved P50 latency.\n";
        }
        else
        {
            std::cout
                << "Fused path did not improve P50 latency; "
                << "retain the Milestone 5 path as the "
                << "performance default until further tuning.\n";
        }

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