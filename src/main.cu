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
    bool fmaTunedKernelUsed = false;
};

struct BatchComparison
{
    int batchSize = 0;

    PathBenchmark milestone11;
    PathBenchmark milestone12;

    GpuQueryTileTuningResult tuning;

    bool m11CpuMatch = false;
    bool m12CpuMatch = false;
    bool m11M12Match = false;

    float m11MaxError = 0.0F;
    float m12MaxError = 0.0F;
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
// Milestone 11 autotuned baseline vs Milestone 12 FMA-tuned
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

    // M11 calibration is outside the measured A/B window.
    // M12 intentionally reuses the same selected tile so
    // this benchmark isolates the instruction-level kernel
    // change rather than conflating two optimizations.
    const GpuQueryTileTuningResult tuning =
        index.autotuneVectorizedQueryTileSize(
            batchQueries,
            batchSize,
            topK,
            7
        );

    index.warmUpTopKBatchTiledVectorizedAuto(
        batchSize,
        topK,
        GPU_WARMUP_RUNS
    );

    index.warmUpTopKBatchTiledVectorizedFmaAuto(
        batchSize,
        topK,
        GPU_WARMUP_RUNS
    );

    std::vector<double> m11TotalTimes;
    std::vector<double> m11KernelTimes;
    std::vector<double> m11MergeTimes;
    std::vector<double> m11UploadTimes;
    std::vector<double> m11DownloadTimes;

    std::vector<double> m12TotalTimes;
    std::vector<double> m12KernelTimes;
    std::vector<double> m12MergeTimes;
    std::vector<double> m12UploadTimes;
    std::vector<double> m12DownloadTimes;

    m11TotalTimes.reserve(BENCHMARK_RUNS);
    m11KernelTimes.reserve(BENCHMARK_RUNS);
    m11MergeTimes.reserve(BENCHMARK_RUNS);
    m11UploadTimes.reserve(BENCHMARK_RUNS);
    m11DownloadTimes.reserve(BENCHMARK_RUNS);

    m12TotalTimes.reserve(BENCHMARK_RUNS);
    m12KernelTimes.reserve(BENCHMARK_RUNS);
    m12MergeTimes.reserve(BENCHMARK_RUNS);
    m12UploadTimes.reserve(BENCHMARK_RUNS);
    m12DownloadTimes.reserve(BENCHMARK_RUNS);

    GpuBatchTopKResult finalM11Result;
    GpuBatchTopKResult finalM12Result;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         ++run)
    {
        GpuBatchTopKResult m11Result;
        GpuBatchTopKResult m12Result;

        if ((run & 1) == 0)
        {
            m11Result =
                index.searchTopKBatchTiledVectorizedAuto(
                    batchQueries,
                    batchSize,
                    topK
                );

            m12Result =
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    batchQueries,
                    batchSize,
                    topK
                );
        }
        else
        {
            m12Result =
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    batchQueries,
                    batchSize,
                    topK
                );

            m11Result =
                index.searchTopKBatchTiledVectorizedAuto(
                    batchQueries,
                    batchSize,
                    topK
                );
        }

        m11TotalTimes.push_back(
            m11Result.totalTimeMs
        );
        m11KernelTimes.push_back(
            static_cast<double>(
                m11Result.fusedKernelTimeMs
            )
        );
        m11MergeTimes.push_back(
            static_cast<double>(
                m11Result.mergeKernelTimeMs
            )
        );
        m11UploadTimes.push_back(
            m11Result.queryUploadTimeMs
        );
        m11DownloadTimes.push_back(
            m11Result.resultDownloadTimeMs
        );

        m12TotalTimes.push_back(
            m12Result.totalTimeMs
        );
        m12KernelTimes.push_back(
            static_cast<double>(
                m12Result.fusedKernelTimeMs
            )
        );
        m12MergeTimes.push_back(
            static_cast<double>(
                m12Result.mergeKernelTimeMs
            )
        );
        m12UploadTimes.push_back(
            m12Result.queryUploadTimeMs
        );
        m12DownloadTimes.push_back(
            m12Result.resultDownloadTimeMs
        );

        if (run ==
            BENCHMARK_RUNS - 1)
        {
            finalM11Result =
                std::move(
                    m11Result
                );

            finalM12Result =
                std::move(
                    m12Result
                );
        }
    }

    BatchComparison comparison;

    comparison.batchSize =
        batchSize;

    comparison.tuning =
        tuning;

    comparison.milestone11.totalStats =
        calculateStats(
            m11TotalTimes
        );
    comparison.milestone11.fusedKernelStats =
        calculateStats(
            m11KernelTimes
        );
    comparison.milestone11.mergeStats =
        calculateStats(
            m11MergeTimes
        );
    comparison.milestone11.uploadStats =
        calculateStats(
            m11UploadTimes
        );
    comparison.milestone11.downloadStats =
        calculateStats(
            m11DownloadTimes
        );
    comparison.milestone11.finalNeighbors =
        std::move(
            finalM11Result.neighbors
        );
    comparison.milestone11.topKPasses =
        finalM11Result.topKPasses;
    comparison.milestone11.queryTileSize =
        finalM11Result.queryTileSize;
    comparison.milestone11.vectorizedLoadsUsed =
        finalM11Result.vectorizedLoadsUsed;
    comparison.milestone11.fmaTunedKernelUsed =
        finalM11Result.fmaTunedKernelUsed;

    comparison.milestone12.totalStats =
        calculateStats(
            m12TotalTimes
        );
    comparison.milestone12.fusedKernelStats =
        calculateStats(
            m12KernelTimes
        );
    comparison.milestone12.mergeStats =
        calculateStats(
            m12MergeTimes
        );
    comparison.milestone12.uploadStats =
        calculateStats(
            m12UploadTimes
        );
    comparison.milestone12.downloadStats =
        calculateStats(
            m12DownloadTimes
        );
    comparison.milestone12.finalNeighbors =
        std::move(
            finalM12Result.neighbors
        );
    comparison.milestone12.topKPasses =
        finalM12Result.topKPasses;
    comparison.milestone12.queryTileSize =
        finalM12Result.queryTileSize;
    comparison.milestone12.vectorizedLoadsUsed =
        finalM12Result.vectorizedLoadsUsed;
    comparison.milestone12.fmaTunedKernelUsed =
        finalM12Result.fmaTunedKernelUsed;

    const std::vector<Neighbor> expected =
        flattenCpuReferences(
            cpuReferences,
            batchSize,
            topK
        );

    comparison.m11CpuMatch =
        neighborIndicesMatch(
            expected,
            comparison
                .milestone11
                .finalNeighbors
        );

    comparison.m12CpuMatch =
        neighborIndicesMatch(
            expected,
            comparison
                .milestone12
                .finalNeighbors
        );

    comparison.m11M12Match =
        neighborIndicesMatch(
            comparison
                .milestone11
                .finalNeighbors,
            comparison
                .milestone12
                .finalNeighbors
        );

    comparison.m11MaxError =
        maxNeighborDistanceError(
            expected,
            comparison
                .milestone11
                .finalNeighbors
        );

    comparison.m12MaxError =
        maxNeighborDistanceError(
            expected,
            comparison
                .milestone12
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
    Milestone10,
    Milestone11,
    Milestone12
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
            << "<vectorCount> <dimension> <topK> <batchSize>\n"
            << "  vortex_cuda --profile-m11 "
            << "<vectorCount> <dimension> <topK> <batchSize>\n"
            << "  vortex_cuda --profile-m12 "
            << "<vectorCount> <dimension> <topK> <batchSize>\n";

        return EXIT_FAILURE;
    }

    const int vectorCount =
        std::stoi(argv[2]);
    const int dimension =
        std::stoi(argv[3]);
    const int topK =
        std::stoi(argv[4]);
    const int batchSize =
        std::stoi(argv[5]);

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

    const bool profileM11 =
        path ==
        ProfilePath::Milestone11;

    const bool profileM12 =
        path ==
        ProfilePath::Milestone12;

    std::cout
        << "============================================\n"
        << (
            profileM12
                ? "       Vortex CUDA M12 Profile Mode\n"
                : (
                    profileM11
                        ? "       Vortex CUDA M11 Profile Mode\n"
                        : (
                            profileM10
                                ? "       Vortex CUDA M10 Profile Mode\n"
                                : "       Vortex CUDA M8 Profile Mode\n"
                          )
                  )
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

    if (profileM10 ||
        profileM11 ||
        profileM12)
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

    if (profileM11 ||
        profileM12)
    {
        std::cout
            << "Autotuning query tile outside profiler range...\n";

        const GpuQueryTileTuningResult tuning =
            gpuIndex.autotuneVectorizedQueryTileSize(
                queries,
                batchSize,
                topK,
                7
            );

        std::cout
            << "Selected query tile: "
            << tuning.selectedTileSize
            << '\n';
    }

    constexpr int PROFILE_WARMUP_RUNS =
        3;

    std::cout
        << "Running warm-ups outside profiler range...\n";

    for (int run = 0;
         run <
             PROFILE_WARMUP_RUNS;
         ++run)
    {
        if (profileM12)
        {
            (void)gpuIndex
                .searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );
        }
        else if (profileM11)
        {
            (void)gpuIndex
                .searchTopKBatchTiledVectorizedAuto(
                    queries,
                    batchSize,
                    topK
                );
        }
        else if (profileM10)
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
            profileM12
                ? "Starting focused M12 profile region...\n"
                : (
                    profileM11
                        ? "Starting focused M11 profile region...\n"
                        : (
                            profileM10
                                ? "Starting focused M10 profile region...\n"
                                : "Starting focused M8 profile region...\n"
                          )
                  )
        );

    CUDA_CHECK(
        cudaProfilerStart()
    );

    GpuBatchTopKResult result;

    if (profileM12)
    {
        result =
            gpuIndex
                .searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );
    }
    else if (profileM11)
    {
        result =
            gpuIndex
                .searchTopKBatchTiledVectorizedAuto(
                    queries,
                    batchSize,
                    topK
                );
    }
    else if (profileM10)
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
        << "FMA-tuned kernel:      "
        << (
            result.fmaTunedKernelUsed
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

            if (mode ==
                "--profile-m11")
            {
                return
                    runProfileMode(
                        argc,
                        argv,
                        ProfilePath::Milestone11
                    );
            }

            if (mode ==
                "--profile-m12")
            {
                return
                    runProfileMode(
                        argc,
                        argv,
                        ProfilePath::Milestone12
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
            << " Vortex CUDA FMA-Tuned Vectorized Search\n"
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
                    : "no - scalar query-tiled fallback"
            )
            << '\n'
            << "Autotune candidates: 1, 2, 4\n"
            << "Autotune trials/tile: 7\n"
            << "M12 arithmetic: explicit RN FMA + short live ranges\n"
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
            const bool m11Numeric =
                std::isfinite(
                    comparison.m11MaxError
                ) &&
                comparison.m11MaxError <=
                    MAX_ALLOWED_ERROR;

            const bool m12Numeric =
                std::isfinite(
                    comparison.m12MaxError
                ) &&
                comparison.m12MaxError <=
                    MAX_ALLOWED_ERROR;

            allCorrect =
                allCorrect &&
                comparison.m11CpuMatch &&
                comparison.m12CpuMatch &&
                comparison.m11M12Match &&
                m11Numeric &&
                m12Numeric;
        }

        std::cout
            << "\n\n============================================\n"
            << "      FMA / Live-Range Tuning Benchmark\n"
            << "============================================\n";

        std::cout
            << "\nBatch Tile  M11 P50  M12 P50  M12 P95"
            << "   M11 QPS   M12 QPS   Gain\n"
            << "------------------------------------------------"
            << "-------------------\n";

        for (const BatchComparison&
             comparison :
             comparisons)
        {
            const double m11Qps =
                static_cast<double>(
                    comparison.batchSize
                ) *
                1000.0 /
                comparison
                    .milestone11
                    .totalStats
                    .p50;

            const double m12Qps =
                static_cast<double>(
                    comparison.batchSize
                ) *
                1000.0 /
                comparison
                    .milestone12
                    .totalStats
                    .p50;

            const double gain =
                comparison
                    .milestone11
                    .totalStats
                    .p50 /
                comparison
                    .milestone12
                    .totalStats
                    .p50;

            std::cout
                << std::setw(5)
                << comparison.batchSize
                << " "
                << std::setw(4)
                << comparison
                    .milestone12
                    .queryTileSize
                << "  "
                << std::setw(7)
                << comparison
                    .milestone11
                    .totalStats
                    .p50
                << "  "
                << std::setw(7)
                << comparison
                    .milestone12
                    .totalStats
                    .p50
                << "  "
                << std::setw(7)
                << comparison
                    .milestone12
                    .totalStats
                    .p95
                << "  "
                << std::setw(8)
                << m11Qps
                << "  "
                << std::setw(8)
                << m12Qps
                << "  "
                << std::setw(5)
                << gain
                << "x\n";
        }

        std::cout
            << "\nM11 Query-Tile Calibration "
            << "(median first-stage kernel ms)\n"
            << "--------------------------------------------\n"
            << "Batch   Tile1   Tile2   Tile4   Selected\n";

        for (const BatchComparison&
             comparison :
             comparisons)
        {
            const auto printCandidate =
                [](
                    const float value)
                {
                    if (value < 0.0F)
                    {
                        std::cout
                            << std::setw(7)
                            << "-";
                    }
                    else
                    {
                        std::cout
                            << std::setw(7)
                            << std::fixed
                            << std::setprecision(3)
                            << value;
                    }
                };

            std::cout
                << std::setw(5)
                << comparison.batchSize
                << " ";

            printCandidate(
                comparison
                    .tuning
                    .tile1MedianKernelTimeMs
            );

            std::cout
                << " ";

            printCandidate(
                comparison
                    .tuning
                    .tile2MedianKernelTimeMs
            );

            std::cout
                << " ";

            printCandidate(
                comparison
                    .tuning
                    .tile4MedianKernelTimeMs
            );

            std::cout
                << "   "
                << comparison
                    .tuning
                    .selectedTileSize
                << '\n';
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
                            .milestone12
                            .totalStats
                            .p50;

                    const double rightQps =
                        static_cast<double>(
                            right.batchSize
                        ) *
                        1000.0 /
                        right
                            .milestone12
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

        const double m11BestQps =
            static_cast<double>(
                best.batchSize
            ) *
            1000.0 /
            best
                .milestone11
                .totalStats
                .p50;

        const double m12BestQps =
            static_cast<double>(
                best.batchSize
            ) *
            1000.0 /
            best
                .milestone12
                .totalStats
                .p50;

        const double bestGain =
            best
                .milestone11
                .totalStats
                .p50 /
            best
                .milestone12
                .totalStats
                .p50;

        const double amortizedP50 =
            best
                .milestone12
                .totalStats
                .p50 /
            static_cast<double>(
                best.batchSize
            );

        const double amortizedP95 =
            best
                .milestone12
                .totalStats
                .p95 /
            static_cast<double>(
                best.batchSize
            );

        std::cout
            << "\nBest M12 Throughput\n"
            << "--------------------------------------------\n"
            << "Batch size:                 "
            << best.batchSize
            << '\n'
            << "Selected query tile:        "
            << best
                .milestone12
                .queryTileSize
            << '\n'
            << "Vectorized loads used:      "
            << (
                best
                    .milestone12
                    .vectorizedLoadsUsed
                    ? "yes"
                    : "no"
            )
            << '\n'
            << "FMA-tuned kernel used:      "
            << (
                best
                    .milestone12
                    .fmaTunedKernelUsed
                    ? "yes"
                    : "no"
            )
            << '\n'
            << "M11 batch P50:              "
            << best
                .milestone11
                .totalStats
                .p50
            << " ms\n"
            << "M12 batch P50:              "
            << best
                .milestone12
                .totalStats
                .p50
            << " ms\n"
            << "M12 batch P95:              "
            << best
                .milestone12
                .totalStats
                .p95
            << " ms\n"
            << "M12 batch P99:              "
            << best
                .milestone12
                .totalStats
                .p99
            << " ms\n"
            << "Amortized M12 P50/query:    "
            << amortizedP50
            << " ms\n"
            << "Amortized M12 P95/query:    "
            << amortizedP95
            << " ms\n"
            << "M11 throughput:             "
            << m11BestQps
            << " queries/sec\n"
            << "M12 throughput:             "
            << m12BestQps
            << " queries/sec\n"
            << "M12 vs M11 throughput gain: "
            << bestGain
            << "x\n"
            << "M11 kernel P50:             "
            << best
                .milestone11
                .fusedKernelStats
                .p50
            << " ms\n"
            << "M12 kernel P50:             "
            << best
                .milestone12
                .fusedKernelStats
                .p50
            << " ms\n"
            << "M12 candidate merge P50:    "
            << best
                .milestone12
                .mergeStats
                .p50
            << " ms\n"
            << "M12 query upload P50:       "
            << best
                .milestone12
                .uploadStats
                .p50
            << " ms\n"
            << "M12 Top-K download P50:     "
            << best
                .milestone12
                .downloadStats
                .p50
            << " ms\n"
            << "Top-K hierarchy passes:     "
            << best
                .milestone12
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
            const bool m11Numeric =
                std::isfinite(
                    comparison.m11MaxError
                ) &&
                comparison.m11MaxError <=
                    MAX_ALLOWED_ERROR;

            const bool m12Numeric =
                std::isfinite(
                    comparison.m12MaxError
                ) &&
                comparison.m12MaxError <=
                    MAX_ALLOWED_ERROR;

            std::cout
                << "Batch "
                << std::setw(2)
                << comparison.batchSize
                << "  M11CPU="
                << (
                    comparison.m11CpuMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  M12CPU="
                << (
                    comparison.m12CpuMatch
                        ? "PASS"
                        : "FAIL"
                )
                << "  M11vsM12="
                << (
                    comparison.m11M12Match
                        ? "PASS"
                        : "FAIL"
                )
                << "  M11Num="
                << (
                    m11Numeric
                        ? "PASS"
                        : "FAIL"
                )
                << "  M12Num="
                << (
                    m12Numeric
                        ? "PASS"
                        : "FAIL"
                )
                << "  M12Error="
                << comparison.m12MaxError
                << '\n';
        }

        if (!allCorrect)
        {
            std::cerr
                << "\nMilestone 12 correctness "
                << "validation failed.\n";

            return EXIT_FAILURE;
        }

        printFirstQueryNeighbors(
            best
                .milestone12
                .finalNeighbors,
            topK
        );

        std::cout
            << "\nMilestone 12 completed successfully.\n";

        if (!best
                .milestone12
                .vectorizedLoadsUsed)
        {
            std::cout
                << "Dimension is not divisible by four; "
                << "M12 correctly preserved the scalar fallback.\n";
        }
        else if (bestGain >=
            1.02)
        {
            std::cout
                << "Instruction/FMA tuning produced a measurable "
                << "throughput improvement.\n";
        }
        else if (bestGain >=
            0.98)
        {
            std::cout
                << "M12 is performance-equivalent to M11 within "
                << "the benchmark noise band; use Nsight counters "
                << "to decide whether to keep the tuned kernel.\n";
        }
        else
        {
            std::cout
                << "M12 is slower than M11 on this workload; "
                << "retain M11 as the performance default.\n";
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
