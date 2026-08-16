#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

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

struct PathBenchmark
{
    TimingStats totalStats;
    TimingStats fusedKernelStats;
    TimingStats mergeStats;
    TimingStats uploadStats;
    TimingStats downloadStats;

    std::vector<Neighbor> finalNeighbors;

    int topKPasses = 0;
    int queryTileSize = 1;
    bool vectorizedLoadsUsed = false;
};

struct BatchComparison
{
    int batchSize = 0;

    PathBenchmark milestone8;
    PathBenchmark milestone10;

    bool m8CpuMatch = false;
    bool m10CpuMatch = false;
    bool m8M10Match = false;

    float m8MaxError = 0.0F;
    float m10MaxError = 0.0F;
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
// CPU exact reference
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
// Query utilities
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
        const std::size_t beginIndex =
            static_cast<std::size_t>(
                queryIndex
            ) *
            static_cast<std::size_t>(
                dimension
            );

        const auto begin =
            queries.begin() +
            static_cast<std::ptrdiff_t>(
                beginIndex
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
    const std::size_t elementCount =
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
                    elementCount
                )
        );
}

std::vector<Neighbor> flattenCpuReferences(
    const std::vector<
        std::vector<Neighbor>
    >& references,
    const int queryCount,
    const int topK)
{
    std::vector<Neighbor> result;

    result.reserve(
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
        const auto& neighbors =
            references[
                static_cast<std::size_t>(
                    queryIndex
                )
            ];

        result.insert(
            result.end(),
            neighbors.begin(),
            neighbors.end()
        );
    }

    return result;
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
// Milestone 8 vs Milestone 10 benchmark
// =========================================================

BatchComparison benchmarkBatchComparison(
    GpuVectorIndex& index,
    const std::vector<float>& allQueries,
    const std::vector<
        std::vector<Neighbor>
    >& cpuReferences,
    const int batchSize,
    const int dimension,
    const int topK)
{
    const std::vector<float> batchQueries =
        prefixQueries(
            allQueries,
            batchSize,
            dimension
        );

    index.warmUpTopKBatchTiled(
        batchSize,
        topK,
        GPU_WARMUP_RUNS
    );

    index.warmUpTopKBatchTiledVectorized(
        batchSize,
        topK,
        GPU_WARMUP_RUNS
    );

    std::vector<double> m8TotalTimes;
    std::vector<double> m8KernelTimes;
    std::vector<double> m8MergeTimes;
    std::vector<double> m8UploadTimes;
    std::vector<double> m8DownloadTimes;

    std::vector<double> m10TotalTimes;
    std::vector<double> m10KernelTimes;
    std::vector<double> m10MergeTimes;
    std::vector<double> m10UploadTimes;
    std::vector<double> m10DownloadTimes;

    m8TotalTimes.reserve(BENCHMARK_RUNS);
    m8KernelTimes.reserve(BENCHMARK_RUNS);
    m8MergeTimes.reserve(BENCHMARK_RUNS);
    m8UploadTimes.reserve(BENCHMARK_RUNS);
    m8DownloadTimes.reserve(BENCHMARK_RUNS);

    m10TotalTimes.reserve(BENCHMARK_RUNS);
    m10KernelTimes.reserve(BENCHMARK_RUNS);
    m10MergeTimes.reserve(BENCHMARK_RUNS);
    m10UploadTimes.reserve(BENCHMARK_RUNS);
    m10DownloadTimes.reserve(BENCHMARK_RUNS);

    GpuBatchTopKResult finalM8Result;
    GpuBatchTopKResult finalM10Result;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuBatchTopKResult m8Result;
        GpuBatchTopKResult m10Result;

        // Alternate order to reduce temperature/order bias.
        if ((run & 1) == 0)
        {
            m8Result =
                index.searchTopKBatchTiled(
                    batchQueries,
                    batchSize,
                    topK
                );

            m10Result =
                index.searchTopKBatchTiledVectorized(
                    batchQueries,
                    batchSize,
                    topK
                );
        }
        else
        {
            m10Result =
                index.searchTopKBatchTiledVectorized(
                    batchQueries,
                    batchSize,
                    topK
                );

            m8Result =
                index.searchTopKBatchTiled(
                    batchQueries,
                    batchSize,
                    topK
                );
        }

        m8TotalTimes.push_back(
            m8Result.totalTimeMs
        );
        m8KernelTimes.push_back(
            static_cast<double>(
                m8Result.fusedKernelTimeMs
            )
        );
        m8MergeTimes.push_back(
            static_cast<double>(
                m8Result.mergeKernelTimeMs
            )
        );
        m8UploadTimes.push_back(
            m8Result.queryUploadTimeMs
        );
        m8DownloadTimes.push_back(
            m8Result.resultDownloadTimeMs
        );

        m10TotalTimes.push_back(
            m10Result.totalTimeMs
        );
        m10KernelTimes.push_back(
            static_cast<double>(
                m10Result.fusedKernelTimeMs
            )
        );
        m10MergeTimes.push_back(
            static_cast<double>(
                m10Result.mergeKernelTimeMs
            )
        );
        m10UploadTimes.push_back(
            m10Result.queryUploadTimeMs
        );
        m10DownloadTimes.push_back(
            m10Result.resultDownloadTimeMs
        );

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            finalM8Result =
                std::move(
                    m8Result
                );

            finalM10Result =
                std::move(
                    m10Result
                );
        }
    }

    BatchComparison comparison;

    comparison.batchSize =
        batchSize;

    comparison.milestone8.totalStats =
        calculateStats(
            m8TotalTimes
        );
    comparison.milestone8.fusedKernelStats =
        calculateStats(
            m8KernelTimes
        );
    comparison.milestone8.mergeStats =
        calculateStats(
            m8MergeTimes
        );
    comparison.milestone8.uploadStats =
        calculateStats(
            m8UploadTimes
        );
    comparison.milestone8.downloadStats =
        calculateStats(
            m8DownloadTimes
        );
    comparison.milestone8.finalNeighbors =
        std::move(
            finalM8Result.neighbors
        );
    comparison.milestone8.topKPasses =
        finalM8Result.topKPasses;
    comparison.milestone8.queryTileSize =
        finalM8Result.queryTileSize;
    comparison.milestone8.vectorizedLoadsUsed =
        finalM8Result.vectorizedLoadsUsed;

    comparison.milestone10.totalStats =
        calculateStats(
            m10TotalTimes
        );
    comparison.milestone10.fusedKernelStats =
        calculateStats(
            m10KernelTimes
        );
    comparison.milestone10.mergeStats =
        calculateStats(
            m10MergeTimes
        );
    comparison.milestone10.uploadStats =
        calculateStats(
            m10UploadTimes
        );
    comparison.milestone10.downloadStats =
        calculateStats(
            m10DownloadTimes
        );
    comparison.milestone10.finalNeighbors =
        std::move(
            finalM10Result.neighbors
        );
    comparison.milestone10.topKPasses =
        finalM10Result.topKPasses;
    comparison.milestone10.queryTileSize =
        finalM10Result.queryTileSize;
    comparison.milestone10.vectorizedLoadsUsed =
        finalM10Result.vectorizedLoadsUsed;

    const std::vector<Neighbor> expected =
        flattenCpuReferences(
            cpuReferences,
            batchSize,
            topK
        );

    comparison.m8CpuMatch =
        neighborIndicesMatch(
            expected,
            comparison
                .milestone8
                .finalNeighbors
        );

    comparison.m10CpuMatch =
        neighborIndicesMatch(
            expected,
            comparison
                .milestone10
                .finalNeighbors
        );

    comparison.m8M10Match =
        neighborIndicesMatch(
            comparison
                .milestone8
                .finalNeighbors,
            comparison
                .milestone10
                .finalNeighbors
        );

    comparison.m8MaxError =
        maxNeighborDistanceError(
            expected,
            comparison
                .milestone8
                .finalNeighbors
        );

    comparison.m10MaxError =
        maxNeighborDistanceError(
            expected,
            comparison
                .milestone10
                .finalNeighbors
        );

    return comparison;
}

// =========================================================
// Printing
// =========================================================

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
// Focused Nsight profiling helpers
// =========================================================

enum class ProfilePath
{
    Milestone8,
    Milestone10
};

int runProfileMode(
    const int argc,
    char** argv,
    const ProfilePath path)
{
    if (argc != 6)
    {
        std::cerr
            << "Invalid profiling arguments.\n"
            << "Usage:\n"
            << "  vortex_cuda --profile-m8 "
            << "<vectorCount> <dimension> <topK> <batchSize>\n"
            << "  vortex_cuda --profile-m10 "
            << "<vectorCount> <dimension> <topK> <batchSize>\n";

        return EXIT_FAILURE;
    }

    const int vectorCount =
        std::stoi(
            argv[2]
        );

    const int dimension =
        std::stoi(
            argv[3]
        );

    const int topK =
        std::stoi(
            argv[4]
        );

    const int batchSize =
        std::stoi(
            argv[5]
        );

    if (vectorCount <= 0 ||
        dimension <= 0 ||
        topK <= 0 ||
        topK > vectorCount ||
        topK >
            GpuVectorIndex::MAX_TOP_K ||
        batchSize <= 0 ||
        batchSize >
            GpuVectorIndex::MAX_BATCH_SIZE)
    {
        std::cerr
            << "Invalid profiling arguments.\n";

        return EXIT_FAILURE;
    }

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

    const std::size_t databaseElements =
        static_cast<std::size_t>(
            vectorCount
        ) *
        static_cast<std::size_t>(
            dimension
        );

    const std::size_t queryElements =
        static_cast<std::size_t>(
            batchSize
        ) *
        static_cast<std::size_t>(
            dimension
        );

    const bool profileM10 =
        path ==
        ProfilePath::Milestone10;

    std::cout
        << "============================================\n"
        << (
            profileM10
                ? "       Vortex CUDA M10 Profile Mode\n"
                : "       Vortex CUDA M8 Profile Mode\n"
        )
        << "============================================\n\n"
        << "GPU: "
        << device.name
        << '\n'
        << "Dataset: "
        << vectorCount
        << " x "
        << dimension
        << '\n'
        << "Top-K: "
        << topK
        << '\n'
        << "Batch size: "
        << batchSize
        << '\n'
        << "Warm-up runs: 3\n";

    if (profileM10)
    {
        std::cout
            << "float4 eligible: "
            << (
                (dimension % 4) == 0
                    ? "yes"
                    : "no - scalar fallback"
            )
            << '\n';
    }

    std::cout
        << '\n'
        << "Generating deterministic profile dataset...\n";

    const std::vector<float> database =
        generateRandomData(
            databaseElements,
            42
        );

    const std::vector<float> queries =
        generateRandomData(
            queryElements,
            1337
        );

    std::cout
        << "Building persistent GPU index...\n";

    GpuVectorIndex gpuIndex(
        database,
        vectorCount,
        dimension
    );

    constexpr int PROFILE_WARMUP_RUNS =
        3;

    std::cout
        << "Running warm-ups outside profiler range...\n";

    for (int run = 0;
         run <
             PROFILE_WARMUP_RUNS;
         ++run)
    {
        if (profileM10)
        {
            (void)gpuIndex
                .searchTopKBatchTiledVectorized(
                    queries,
                    batchSize,
                    topK
                );
        }
        else
        {
            (void)gpuIndex
                .searchTopKBatchTiled(
                    queries,
                    batchSize,
                    topK
                );
        }
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::cout
        << (
            profileM10
                ? "Starting focused M10 profile region...\n"
                : "Starting focused M8 profile region...\n"
        );

    CUDA_CHECK(
        cudaProfilerStart()
    );

    GpuBatchTopKResult result;

    if (profileM10)
    {
        result =
            gpuIndex
                .searchTopKBatchTiledVectorized(
                    queries,
                    batchSize,
                    topK
                );
    }
    else
    {
        result =
            gpuIndex
                .searchTopKBatchTiled(
                    queries,
                    batchSize,
                    topK
                );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaProfilerStop()
    );

    std::cout
        << std::fixed
        << std::setprecision(3)
        << "\nFocused profile run completed.\n"
        << "Query tile size:       "
        << result.queryTileSize
        << '\n'
        << "Vectorized loads:      "
        << (
            result.vectorizedLoadsUsed
                ? "yes"
                : "no"
        )
        << '\n'
        << "Tiled kernel time:     "
        << result.fusedKernelTimeMs
        << " ms\n"
        << "Candidate merge time:  "
        << result.mergeKernelTimeMs
        << " ms\n"
        << "End-to-end time:       "
        << result.totalTimeMs
        << " ms\n"
        << "Returned neighbors:    "
        << result.neighbors.size()
        << '\n';

    return EXIT_SUCCESS;
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
        if (argc >= 2)
        {
            const std::string mode(
                argv[1]
            );

            if (mode ==
                "--profile-m8")
            {
                return
                    runProfileMode(
                        argc,
                        argv,
                        ProfilePath::Milestone8
                    );
            }

            if (mode ==
                "--profile-m10")
            {
                return
                    runProfileMode(
                        argc,
                        argv,
                        ProfilePath::Milestone10
                    );
            }
        }

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
            << " Vortex CUDA Vectorized Query-Tiled Search\n"
            << "============================================\n\n"
            << "GPU: "
            << device.name
            << '\n'
            << "Compute capability: "
            << device.major
            << "."
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
            << "Max query tile: "
            << GpuVectorIndex::MAX_QUERY_TILE_SIZE
            << '\n'
            << "float4 eligible: "
            << (
                (dimension % 4) == 0
                    ? "yes"
                    : "no - M10 uses M8 fallback"
            )
            << '\n'
            << "Benchmark runs/path: "
            << BENCHMARK_RUNS
            << '\n'
            << "A/B ordering: alternating\n\n";

        const std::size_t databaseElements =
            static_cast<std::size_t>(
                vectorCount
            ) *
            static_cast<std::size_t>(
                dimension
            );

        std::cout
            << "Generating deterministic database...\n";

        const std::vector<float> database =
            generateRandomData(
                databaseElements,
                42
            );

        std::cout
            << "Generating "
            << GpuVectorIndex::MAX_BATCH_SIZE
            << " deterministic queries...\n";

        const std::vector<float> allQueries =
            generateRandomData(
                static_cast<std::size_t>(
                    GpuVectorIndex::MAX_BATCH_SIZE
                ) *
                static_cast<std::size_t>(
                    dimension
                ),
                1337
            );

        const std::vector<std::vector<float>>
            splitQueryVectors =
                splitQueries(
                    allQueries,
                    GpuVectorIndex::MAX_BATCH_SIZE,
                    dimension
                );

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

        std::cout
            << "\nBuilding CPU correctness references "
            << "for all "
            << GpuVectorIndex::MAX_BATCH_SIZE
            << " queries...\n";

        std::vector<
            std::vector<Neighbor>
        > cpuReferences;

        cpuReferences.reserve(
            GpuVectorIndex::MAX_BATCH_SIZE
        );

        for (int queryIndex = 0;
             queryIndex <
                 GpuVectorIndex::MAX_BATCH_SIZE;
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

        std::vector<BatchComparison>
            comparisons;

        comparisons.reserve(
            BATCH_SIZES.size()
        );

        for (const int batchSize :
             BATCH_SIZES)
        {
            std::cout
                << "Benchmarking batch size "
                << batchSize
                << "...\n";

            comparisons.push_back(
                benchmarkBatchComparison(
                    gpuIndex,
                    allQueries,
                    cpuReferences,
                    batchSize,
                    dimension,
                    topK
                )
            );
        }

        bool allCorrect =
            true;

        for (const BatchComparison&
             comparison :
             comparisons)
        {
            const bool m8Numeric =
                std::isfinite(
                    comparison.m8MaxError
                ) &&
                comparison.m8MaxError <=
                    MAX_ALLOWED_ERROR;

            const bool m10Numeric =
                std::isfinite(
                    comparison.m10MaxError
                ) &&
                comparison.m10MaxError <=
                    MAX_ALLOWED_ERROR;

            allCorrect =
                allCorrect &&
                comparison.m8CpuMatch &&
                comparison.m10CpuMatch &&
                comparison.m8M10Match &&
                m8Numeric &&
                m10Numeric;
        }

        std::cout
            << "\n\n============================================\n"
            << "       Vectorized Batch Benchmark Results\n"
            << "============================================\n";

        std::cout
            << "\nBatch Tile Vec  M8 P50  M10 P50  M10 P95"
            << "   M8 QPS   M10 QPS   Gain\n"
            << "------------------------------------------------"
            << "-----------------------\n";

        for (const BatchComparison&
             comparison :
             comparisons)
        {
            const double m8Qps =
                static_cast<double>(
                    comparison.batchSize
                ) *
                1000.0 /
                comparison
                    .milestone8
                    .totalStats
                    .p50;

            const double m10Qps =
                static_cast<double>(
                    comparison.batchSize
                ) *
                1000.0 /
                comparison
                    .milestone10
                    .totalStats
                    .p50;

            const double gain =
                comparison
                    .milestone8
                    .totalStats
                    .p50 /
                comparison
                    .milestone10
                    .totalStats
                    .p50;

            std::cout
                << std::setw(5)
                << comparison.batchSize
                << " "
                << std::setw(4)
                << comparison
                    .milestone10
                    .queryTileSize
                << " "
                << std::setw(3)
                << (
                    comparison
                        .milestone10
                        .vectorizedLoadsUsed
                        ? "yes"
                        : "no"
                )
                << "  "
                << std::setw(7)
                << comparison
                    .milestone8
                    .totalStats
                    .p50
                << "  "
                << std::setw(7)
                << comparison
                    .milestone10
                    .totalStats
                    .p50
                << "  "
                << std::setw(7)
                << comparison
                    .milestone10
                    .totalStats
                    .p95
                << "  "
                << std::setw(8)
                << m8Qps
                << "  "
                << std::setw(8)
                << m10Qps
                << "  "
                << std::setw(5)
                << gain
                << "x\n";
        }

        const auto bestIterator =
            std::max_element(
                comparisons.begin(),
                comparisons.end(),
                [](
                    const BatchComparison& left,
                    const BatchComparison& right)
                {
                    const double leftQps =
                        static_cast<double>(
                            left.batchSize
                        ) *
                        1000.0 /
                        left
                            .milestone10
                            .totalStats
                            .p50;

                    const double rightQps =
                        static_cast<double>(
                            right.batchSize
                        ) *
                        1000.0 /
                        right
                            .milestone10
                            .totalStats
                            .p50;

                    return
                        leftQps <
                        rightQps;
                }
            );

        if (bestIterator ==
            comparisons.end())
        {
            throw std::runtime_error(
                "No benchmark results."
            );
        }

        const BatchComparison& best =
            *bestIterator;

        const double m8BestQps =
            static_cast<double>(
                best.batchSize
            ) *
            1000.0 /
            best
                .milestone8
                .totalStats
                .p50;

        const double m10BestQps =
            static_cast<double>(
                best.batchSize
            ) *
            1000.0 /
            best
                .milestone10
                .totalStats
                .p50;

        const double bestGain =
            best
                .milestone8
                .totalStats
                .p50 /
            best
                .milestone10
                .totalStats
                .p50;

        const double amortizedP50 =
            best
                .milestone10
                .totalStats
                .p50 /
            static_cast<double>(
                best.batchSize
            );

        const double amortizedP95 =
            best
                .milestone10
                .totalStats
                .p95 /
            static_cast<double>(
                best.batchSize
            );

        std::cout
            << "\nBest M10 Throughput\n"
            << "--------------------------------------------\n"
            << "Batch size:                 "
            << best.batchSize
            << '\n'
            << "Query tile size:            "
            << best
                .milestone10
                .queryTileSize
            << '\n'
            << "Vectorized loads used:      "
            << (
                best
                    .milestone10
                    .vectorizedLoadsUsed
                    ? "yes"
                    : "no"
            )
            << '\n'
            << "M8 batch P50:               "
            << best
                .milestone8
                .totalStats
                .p50
            << " ms\n"
            << "M10 batch P50:              "
            << best
                .milestone10
                .totalStats
                .p50
            << " ms\n"
            << "M10 batch P95:              "
            << best
                .milestone10
                .totalStats
                .p95
            << " ms\n"
            << "M10 batch P99:              "
            << best
                .milestone10
                .totalStats
                .p99
            << " ms\n"
            << "Amortized M10 P50/query:    "
            << amortizedP50
            << " ms\n"
            << "Amortized M10 P95/query:    "
            << amortizedP95
            << " ms\n"
            << "M8 throughput:              "
            << m8BestQps
            << " queries/sec\n"
            << "M10 throughput:             "
            << m10BestQps
            << " queries/sec\n"
            << "M10 vs M8 throughput gain:  "
            << bestGain
            << "x\n"
            << "M8 tiled kernel P50:        "
            << best
                .milestone8
                .fusedKernelStats
                .p50
            << " ms\n"
            << "M10 float4 kernel P50:      "
            << best
                .milestone10
                .fusedKernelStats
                .p50
            << " ms\n"
            << "M10 candidate merge P50:    "
            << best
                .milestone10
                .mergeStats
                .p50
            << " ms\n"
            << "M10 query upload P50:       "
            << best
                .milestone10
                .uploadStats
                .p50
            << " ms\n"
            << "M10 Top-K download P50:     "
            << best
                .milestone10
                .downloadStats
                .p50
            << " ms\n"
            << "Top-K hierarchy passes:     "
            << best
                .milestone10
                .topKPasses
            << '\n';

        std::cout
            << "\nCorrectness\n"
            << "--------------------------------------------\n"
            << std::setprecision(8);

        for (const BatchComparison&
             comparison :
             comparisons)
        {
            const bool m8Numeric =
                std::isfinite(
                    comparison.m8MaxError
                ) &&
                comparison.m8MaxError <=
                    MAX_ALLOWED_ERROR;

            const bool m10Numeric =
                std::isfinite(
                    comparison.m10MaxError
                ) &&
                comparison.m10MaxError <=
                    MAX_ALLOWED_ERROR;

            std::cout
                << "Batch "
                << std::setw(2)
                << comparison.batchSize
                << "  M8CPU="
                << (
                    comparison.m8CpuMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  M10CPU="
                << (
                    comparison.m10CpuMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  M8vsM10="
                << (
                    comparison.m8M10Match
                        ? "PASS"
                        : "FAIL"
                )
                << "  M8Num="
                << (
                    m8Numeric
                        ? "PASS"
                        : "FAIL"
                )
                << "  M10Num="
                << (
                    m10Numeric
                        ? "PASS"
                        : "FAIL"
                )
                << "  M10Error="
                << comparison.m10MaxError
                << '\n';
        }

        if (!allCorrect)
        {
            std::cerr
                << "\nMilestone 10 correctness "
                << "validation failed.\n";

            return EXIT_FAILURE;
        }

        printFirstQueryNeighbors(
            best
                .milestone10
                .finalNeighbors,
            topK
        );

        std::cout
            << "\nMilestone 10 completed successfully.\n";

        if (!best
                .milestone10
                .vectorizedLoadsUsed)
        {
            std::cout
                << "Dimension is not divisible by four; "
                << "M10 correctly used the M8 scalar fallback.\n";
        }
        else if (bestGain >
            1.0)
        {
            std::cout
                << "float4 vectorization improved measured "
                << "batch throughput.\n";
        }
        else
        {
            std::cout
                << "float4 vectorization did not improve "
                << "measured throughput; retain M8 as the "
                << "performance default until profiling "
                << "explains the result.\n";
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

