#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "gpu_vector_index.cuh"

#include <algorithm>
#include <array>
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

constexpr int GPU_WARMUP_RUNS =
    5;

constexpr int BENCHMARK_RUNS =
    50;

constexpr float MAX_ALLOWED_ERROR =
    0.001F;

constexpr std::array<int, 6>
    BATCH_SIZES{
        1,
        2,
        4,
        8,
        16,
        32
    };

static_assert(
    BATCH_SIZES.back() <=
    GpuVectorIndex::MAX_BATCH_SIZE
);

struct TimingStats
{
    double minimum = 0.0;
    double average = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

struct SequentialBatchRun
{
    double totalTimeMs = 0.0;
    std::vector<Neighbor> neighbors;
};

struct BatchBenchmark
{
    int batchSize = 0;

    TimingStats batchedTotalStats;
    TimingStats sequentialTotalStats;

    TimingStats fusedKernelStats;
    TimingStats mergeStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<Neighbor> finalBatchNeighbors;
    std::vector<Neighbor> finalSequentialNeighbors;

    int topKPasses = 0;

    bool cpuMatch = false;
    bool sequentialMatch = false;

    float maxDistanceError = 0.0F;
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
// CPU exact distance
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

std::vector<Neighbor> searchCpuExact(
    const std::vector<float>& database,
    const std::vector<float>& query,
    const int vectorCount,
    const int dimension,
    const int topK)
{
    return
        findTopKNeighbors(
            calculateDistancesCpu(
                database,
                query,
                vectorCount,
                dimension
            ),
            topK
        );
}

// =========================================================
// Query handling
// =========================================================

std::vector<std::vector<float>>
splitQueries(
    const std::vector<float>& queries,
    const int queryCount,
    const int dimension)
{
    std::vector<std::vector<float>>
        result;

    result.reserve(
        static_cast<std::size_t>(
            queryCount
        )
    );

    for (int queryIndex = 0;
         queryIndex < queryCount;
         ++queryIndex)
    {
        const auto begin =
            queries.begin() +
            static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(
                    queryIndex
                ) *
                static_cast<std::size_t>(
                    dimension
                )
            );

        const auto end =
            begin +
            dimension;

        result.emplace_back(
            begin,
            end
        );
    }

    return result;
}

std::vector<float> prefixQueries(
    const std::vector<float>& queries,
    const int queryCount,
    const int dimension)
{
    const std::size_t count =
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            dimension
        );

    return
        std::vector<float>(
            queries.begin(),
            queries.begin() +
                static_cast<std::ptrdiff_t>(
                    count
                )
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
// Sequential fused batch baseline
// =========================================================

SequentialBatchRun runSequentialBatch(
    GpuVectorIndex& index,
    const std::vector<std::vector<float>>& queries,
    const int queryCount,
    const int topK,
    const bool captureNeighbors)
{
    SequentialBatchRun run;

    if (captureNeighbors)
    {
        run.neighbors.reserve(
            static_cast<std::size_t>(
                queryCount
            ) *
            static_cast<std::size_t>(
                topK
            )
        );
    }

    for (int queryIndex = 0;
         queryIndex < queryCount;
         ++queryIndex)
    {
        GpuFusedTopKResult result =
            index.searchTopKFused(
                queries[
                    static_cast<std::size_t>(
                        queryIndex
                    )
                ],
                topK
            );

        run.totalTimeMs +=
            result.totalTimeMs;

        if (captureNeighbors)
        {
            run.neighbors.insert(
                run.neighbors.end(),
                result.neighbors.begin(),
                result.neighbors.end()
            );
        }
    }

    return run;
}

// =========================================================
// Correctness
// =========================================================

std::vector<Neighbor> flattenCpuReferences(
    const std::vector<
        std::vector<Neighbor>
    >& references,
    const int queryCount,
    const int topK)
{
    std::vector<Neighbor> flattened;

    flattened.reserve(
        static_cast<std::size_t>(
            queryCount
        ) *
        static_cast<std::size_t>(
            topK
        )
    );

    for (int queryIndex = 0;
         queryIndex < queryCount;
         ++queryIndex)
    {
        const auto& queryNeighbors =
            references[
                static_cast<std::size_t>(
                    queryIndex
                )
            ];

        flattened.insert(
            flattened.end(),
            queryNeighbors.begin(),
            queryNeighbors.end()
        );
    }

    return flattened;
}

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

    float maxError = 0.0F;

    for (std::size_t index = 0;
         index < expected.size();
         ++index)
    {
        maxError =
            std::max(
                maxError,
                std::abs(
                    expected[index].distance -
                    actual[index].distance
                )
            );
    }

    return maxError;
}

// =========================================================
// Batch benchmark
// =========================================================

BatchBenchmark benchmarkBatch(
    GpuVectorIndex& index,
    const std::vector<float>& contiguousQueries,
    const std::vector<std::vector<float>>& splitQueryVectors,
    const std::vector<
        std::vector<Neighbor>
    >& cpuReferences,
    const int batchSize,
    const int dimension,
    const int topK)
{
    const std::vector<float> batchQueries =
        prefixQueries(
            contiguousQueries,
            batchSize,
            dimension
        );

    // -----------------------------------------------------
    // Warm both paths.
    // -----------------------------------------------------

    index.warmUpTopKBatch(
        batchSize,
        topK,
        GPU_WARMUP_RUNS
    );

    for (int warmup = 0;
         warmup < GPU_WARMUP_RUNS;
         ++warmup)
    {
        (void)runSequentialBatch(
            index,
            splitQueryVectors,
            batchSize,
            topK,
            false
        );
    }

    std::vector<double>
        batchedTotalTimes;

    std::vector<double>
        sequentialTotalTimes;

    std::vector<double>
        fusedKernelTimes;

    std::vector<double>
        mergeTimes;

    std::vector<double>
        uploadTimes;

    std::vector<double>
        downloadTimes;

    batchedTotalTimes.reserve(
        BENCHMARK_RUNS
    );

    sequentialTotalTimes.reserve(
        BENCHMARK_RUNS
    );

    fusedKernelTimes.reserve(
        BENCHMARK_RUNS
    );

    mergeTimes.reserve(
        BENCHMARK_RUNS
    );

    uploadTimes.reserve(
        BENCHMARK_RUNS
    );

    downloadTimes.reserve(
        BENCHMARK_RUNS
    );

    std::vector<Neighbor>
        finalBatchNeighbors;

    std::vector<Neighbor>
        finalSequentialNeighbors;

    int topKPasses = 0;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuBatchTopKResult batchResult;
        SequentialBatchRun sequentialResult;

        const bool capture =
            run ==
            BENCHMARK_RUNS - 1;

        // Alternate A/B order to reduce
        // systematic order and temperature bias.
        if ((run & 1) == 0)
        {
            batchResult =
                index.searchTopKBatch(
                    batchQueries,
                    batchSize,
                    topK
                );

            sequentialResult =
                runSequentialBatch(
                    index,
                    splitQueryVectors,
                    batchSize,
                    topK,
                    capture
                );
        }
        else
        {
            sequentialResult =
                runSequentialBatch(
                    index,
                    splitQueryVectors,
                    batchSize,
                    topK,
                    capture
                );

            batchResult =
                index.searchTopKBatch(
                    batchQueries,
                    batchSize,
                    topK
                );
        }

        batchedTotalTimes.push_back(
            batchResult.totalTimeMs
        );

        sequentialTotalTimes.push_back(
            sequentialResult.totalTimeMs
        );

        fusedKernelTimes.push_back(
            static_cast<double>(
                batchResult.fusedKernelTimeMs
            )
        );

        mergeTimes.push_back(
            static_cast<double>(
                batchResult.mergeKernelTimeMs
            )
        );

        uploadTimes.push_back(
            batchResult.queryUploadTimeMs
        );

        downloadTimes.push_back(
            batchResult.resultDownloadTimeMs
        );

        topKPasses =
            batchResult.topKPasses;

        if (capture)
        {
            finalBatchNeighbors =
                std::move(
                    batchResult.neighbors
                );

            finalSequentialNeighbors =
                std::move(
                    sequentialResult.neighbors
                );
        }
    }

    BatchBenchmark benchmark;

    benchmark.batchSize =
        batchSize;

    benchmark.batchedTotalStats =
        calculateStats(
            batchedTotalTimes
        );

    benchmark.sequentialTotalStats =
        calculateStats(
            sequentialTotalTimes
        );

    benchmark.fusedKernelStats =
        calculateStats(
            fusedKernelTimes
        );

    benchmark.mergeStats =
        calculateStats(
            mergeTimes
        );

    benchmark.uploadStats =
        calculateStats(
            uploadTimes
        );

    benchmark.downloadStats =
        calculateStats(
            downloadTimes
        );

    benchmark.finalBatchNeighbors =
        std::move(
            finalBatchNeighbors
        );

    benchmark.finalSequentialNeighbors =
        std::move(
            finalSequentialNeighbors
        );

    benchmark.topKPasses =
        topKPasses;

    const std::vector<Neighbor>
        expected =
            flattenCpuReferences(
                cpuReferences,
                batchSize,
                topK
            );

    benchmark.cpuMatch =
        neighborIndicesMatch(
            expected,
            benchmark.finalBatchNeighbors
        );

    benchmark.sequentialMatch =
        neighborIndicesMatch(
            benchmark.finalSequentialNeighbors,
            benchmark.finalBatchNeighbors
        );

    benchmark.maxDistanceError =
        maxNeighborDistanceError(
            expected,
            benchmark.finalBatchNeighbors
        );

    return benchmark;
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
        << '\n'
        << "  Min:       "
        << stats.minimum
        << " ms\n"
        << "  Average:   "
        << stats.average
        << " ms\n"
        << "  P50:       "
        << stats.p50
        << " ms\n"
        << "  P95:       "
        << stats.p95
        << " ms\n"
        << "  P99:       "
        << stats.p99
        << " ms\n";
}

void printFirstQueryNeighbors(
    const std::vector<Neighbor>& neighbors,
    const int topK)
{
    std::cout
        << "\nQuery 0 Top-K\n"
        << "--------------------------------------------\n";

    for (int rank = 0;
         rank < topK;
         ++rank)
    {
        const Neighbor& neighbor =
            neighbors[
                static_cast<std::size_t>(
                    rank
                )
            ];

        std::cout
            << "#"
            << std::setw(2)
            << rank + 1
            << "  vector="
            << std::setw(7)
            << neighbor.index
            << "  squared_l2="
            << std::fixed
            << std::setprecision(6)
            << neighbor.distance
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
                << "[topK]\n";

            return EXIT_FAILURE;
        }

        // -------------------------------------------------
        // CUDA
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
                << "No CUDA GPU detected.\n";

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
            << "       Vortex CUDA Batched Exact Search\n"
            << "============================================\n\n"
            << "GPU: "
            << device.name
            << '\n'
            << "Compute capability: "
            << device.major
            << '.'
            << device.minor
            << '\n'
            << "Multiprocessors: "
            << device.multiProcessorCount
            << '\n'
            << "Dataset: "
            << vectorCount
            << " x "
            << dimension
            << '\n'
            << "Top-K: "
            << topK
            << '\n'
            << "Max batch size: "
            << GpuVectorIndex::MAX_BATCH_SIZE
            << '\n'
            << "Benchmark runs/batch: "
            << BENCHMARK_RUNS
            << '\n'
            << "A/B ordering: alternating"
            << "\n\n";

        // -------------------------------------------------
        // Dataset
        // -------------------------------------------------

        std::cout
            << "Generating deterministic database...\n";

        const std::vector<float> database =
            generateRandomData(
                static_cast<std::size_t>(
                    vectorCount
                ) *
                static_cast<std::size_t>(
                    dimension
                ),
                42
            );

        std::cout
            << "Generating "
            << GpuVectorIndex::MAX_BATCH_SIZE
            << " deterministic queries...\n";

        const std::vector<float> allQueries =
            generateRandomData(
                static_cast<std::size_t>(
                    GpuVectorIndex::
                        MAX_BATCH_SIZE
                ) *
                static_cast<std::size_t>(
                    dimension
                ),
                1337
            );

        const auto splitQueryVectors =
            splitQueries(
                allQueries,
                GpuVectorIndex::
                    MAX_BATCH_SIZE,
                dimension
            );

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
            << std::setprecision(3)
            << "Index size:       "
            << indexSizeMiB
            << " MiB\n"
            << "One-time upload:  "
            << gpuIndex.indexUploadTimeMs()
            << " ms\n";

        // -------------------------------------------------
        // CPU correctness references.
        //
        // Only one CPU execution/query is required here.
        // CPU performance was benchmarked in Milestone 6.
        // -------------------------------------------------

        std::cout
            << "\nBuilding CPU correctness references "
            << "for all "
            << GpuVectorIndex::MAX_BATCH_SIZE
            << " queries...\n";

        std::vector<
            std::vector<Neighbor>
        > cpuReferences;

        cpuReferences.reserve(
            GpuVectorIndex::
                MAX_BATCH_SIZE
        );

        for (int queryIndex = 0;
             queryIndex <
                 GpuVectorIndex::
                     MAX_BATCH_SIZE;
             ++queryIndex)
        {
            cpuReferences.push_back(
                searchCpuExact(
                    database,
                    splitQueryVectors[
                        static_cast<std::size_t>(
                            queryIndex
                        )
                    ],
                    vectorCount,
                    dimension,
                    topK
                )
            );
        }

        // -------------------------------------------------
        // Benchmark all batch sizes.
        // -------------------------------------------------

        std::vector<BatchBenchmark>
            benchmarks;

        benchmarks.reserve(
            BATCH_SIZES.size()
        );

        for (const int batchSize :
             BATCH_SIZES)
        {
            std::cout
                << "Benchmarking batch size "
                << batchSize
                << "...\n";

            benchmarks.push_back(
                benchmarkBatch(
                    gpuIndex,
                    allQueries,
                    splitQueryVectors,
                    cpuReferences,
                    batchSize,
                    dimension,
                    topK
                )
            );
        }

        // -------------------------------------------------
        // Validate all batch sizes.
        // -------------------------------------------------

        bool allCorrect = true;

        for (const BatchBenchmark&
             benchmark :
             benchmarks)
        {
            const bool numericPass =
                std::isfinite(
                    benchmark.maxDistanceError
                ) &&
                benchmark.maxDistanceError <=
                    MAX_ALLOWED_ERROR;

            allCorrect =
                allCorrect &&
                benchmark.cpuMatch &&
                benchmark.sequentialMatch &&
                numericPass;
        }

        // -------------------------------------------------
        // Results
        // -------------------------------------------------

        std::cout
            << "\n\n============================================\n"
            << "            Batch Benchmark Results\n"
            << "============================================\n";

        std::cout
            << "\nBatch   Batch P50   Batch P95   "
            << "ms/query   QPS       Seq P50   Gain\n"
            << "------------------------------------------------"
            << "----------------\n";

        for (const BatchBenchmark&
             benchmark :
             benchmarks)
        {
            const double
                amortizedLatency =
                    benchmark
                        .batchedTotalStats
                        .p50 /
                    static_cast<double>(
                        benchmark.batchSize
                    );

            const double qps =
                static_cast<double>(
                    benchmark.batchSize
                ) *
                1000.0 /
                benchmark
                    .batchedTotalStats
                    .p50;

            const double gain =
                benchmark
                    .sequentialTotalStats
                    .p50 /
                benchmark
                    .batchedTotalStats
                    .p50;

            std::cout
                << std::setw(5)
                << benchmark.batchSize
                << "   "
                << std::setw(9)
                << benchmark
                    .batchedTotalStats
                    .p50
                << "   "
                << std::setw(9)
                << benchmark
                    .batchedTotalStats
                    .p95
                << "   "
                << std::setw(8)
                << amortizedLatency
                << "   "
                << std::setw(8)
                << qps
                << "   "
                << std::setw(8)
                << benchmark
                    .sequentialTotalStats
                    .p50
                << "   "
                << std::setw(5)
                << gain
                << "x\n";
        }

        // -------------------------------------------------
        // Best throughput configuration.
        // -------------------------------------------------

        const auto bestIterator =
            std::max_element(
                benchmarks.begin(),
                benchmarks.end(),
                [](
                    const BatchBenchmark& left,
                    const BatchBenchmark& right)
                {
                    const double leftQps =
                        static_cast<double>(
                            left.batchSize
                        ) *
                        1000.0 /
                        left.batchedTotalStats.p50;

                    const double rightQps =
                        static_cast<double>(
                            right.batchSize
                        ) *
                        1000.0 /
                        right.batchedTotalStats.p50;

                    return
                        leftQps <
                        rightQps;
                }
            );

        if (bestIterator ==
            benchmarks.end())
        {
            throw std::runtime_error(
                "No batch benchmark results."
            );
        }

        const BatchBenchmark&
            bestBenchmark =
                *bestIterator;

        const double bestQps =
            static_cast<double>(
                bestBenchmark.batchSize
            ) *
            1000.0 /
            bestBenchmark
                .batchedTotalStats
                .p50;

        const double bestAmortizedP50 =
            bestBenchmark
                .batchedTotalStats
                .p50 /
            static_cast<double>(
                bestBenchmark.batchSize
            );

        const double bestAmortizedP95 =
            bestBenchmark
                .batchedTotalStats
                .p95 /
            static_cast<double>(
                bestBenchmark.batchSize
            );

        const double bestGain =
            bestBenchmark
                .sequentialTotalStats
                .p50 /
            bestBenchmark
                .batchedTotalStats
                .p50;

        std::cout
            << "\nBest Throughput Configuration\n"
            << "--------------------------------------------\n"
            << "Batch size:                 "
            << bestBenchmark.batchSize
            << '\n'
            << "Batch P50:                  "
            << bestBenchmark
                .batchedTotalStats
                .p50
            << " ms\n"
            << "Batch P95:                  "
            << bestBenchmark
                .batchedTotalStats
                .p95
            << " ms\n"
            << "Amortized P50/query:        "
            << bestAmortizedP50
            << " ms\n"
            << "Amortized P95/query:        "
            << bestAmortizedP95
            << " ms\n"
            << "Throughput:                 "
            << bestQps
            << " queries/sec\n"
            << "Sequential same-work P50:  "
            << bestBenchmark
                .sequentialTotalStats
                .p50
            << " ms\n"
            << "Batch throughput gain:      "
            << bestGain
            << "x\n"
            << "Fused batch kernel P50:     "
            << bestBenchmark
                .fusedKernelStats
                .p50
            << " ms\n"
            << "Candidate merge P50:        "
            << bestBenchmark
                .mergeStats
                .p50
            << " ms\n"
            << "Query upload P50:           "
            << bestBenchmark
                .uploadStats
                .p50
            << " ms\n"
            << "Top-K download P50:         "
            << bestBenchmark
                .downloadStats
                .p50
            << " ms\n"
            << "Top-K hierarchy passes:     "
            << bestBenchmark
                .topKPasses
            << '\n';

        // -------------------------------------------------
        // Correctness
        // -------------------------------------------------

        std::cout
            << "\nCorrectness\n"
            << "--------------------------------------------\n"
            << std::setprecision(8);

        for (const BatchBenchmark&
             benchmark :
             benchmarks)
        {
            const bool numericPass =
                std::isfinite(
                    benchmark.maxDistanceError
                ) &&
                benchmark.maxDistanceError <=
                    MAX_ALLOWED_ERROR;

            std::cout
                << "Batch "
                << std::setw(2)
                << benchmark.batchSize
                << "  CPU="
                << (
                    benchmark.cpuMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  Sequential="
                << (
                    benchmark.sequentialMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  Numeric="
                << (
                    numericPass
                        ? "PASS"
                        : "FAIL"
                )
                << "  maxError="
                << benchmark.maxDistanceError
                << '\n';
        }

        if (!allCorrect)
        {
            std::cerr
                << "\nMilestone 7 correctness "
                << "validation failed.\n";

            return EXIT_FAILURE;
        }

        printFirstQueryNeighbors(
            bestBenchmark
                .finalBatchNeighbors,
            topK
        );

        std::cout
            << "\nMilestone 7 completed successfully.\n";

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