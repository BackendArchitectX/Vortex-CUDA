#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "gpu_vector_index.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef VORTEX_GIT_COMMIT
#define VORTEX_GIT_COMMIT "unknown"
#endif

#ifndef VORTEX_GIT_DIRTY
#define VORTEX_GIT_DIRTY 1
#endif

namespace
{

constexpr unsigned int DATABASE_SEED = 42U;
constexpr unsigned int QUERY_SEED = 1337U;
constexpr int AUTOTUNE_TRIALS = 7;
constexpr double MIN_RECALL_AT_K = 0.98;

struct TimingStats
{
    double minimum = 0.0;
    double mean = 0.0;
    double standardDeviation = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

struct BenchmarkPreset
{
    std::string name;
    std::vector<int> vectorCounts;
    std::vector<int> dimensions;
    std::vector<int> topKs;
    std::vector<int> batchSizes;
    int warmupRuns = 0;
    int measuredRuns = 0;
};

struct RawSample
{
    std::string runId;
    std::string preset;
    int vectorCount = 0;
    int dimension = 0;
    int topK = 0;
    int batchSize = 0;
    int runIndex = 0;
    std::string ordering;
    std::string path;
    int queryTileSize = 1;
    bool vectorizedLoadsUsed = false;
    bool fmaTunedKernelUsed = false;
    bool mixedPrecisionUsed = false;
    bool half2LoadsUsed = false;
    double totalTimeMs = 0.0;
    double fusedKernelTimeMs = 0.0;
    double mergeKernelTimeMs = 0.0;
    double queryUploadTimeMs = 0.0;
    double resultDownloadTimeMs = 0.0;
};

struct CaseSummary
{
    int vectorCount = 0;
    int dimension = 0;
    int topK = 0;
    int batchSize = 0;
    int queryTileSize = 1;
    bool half2LoadsUsed = false;

    std::size_t fp32IndexBytes = 0;
    std::size_t fp16IndexBytes = 0;
    double memoryReductionPercent = 0.0;
    double fp32IndexUploadTimeMs = 0.0;
    double fp16IndexBuildTimeMs = 0.0;

    TimingStats fp32Total;
    TimingStats fp16Total;
    TimingStats fp32Kernel;
    TimingStats fp16Kernel;

    double fp32Qps = 0.0;
    double fp16Qps = 0.0;
    double throughputGain = 0.0;

    double recallAt1 = 0.0;
    double recallAtK = 0.0;
    double meanMatchedDistanceError = 0.0;
    double maxMatchedDistanceError = 0.0;
    bool finiteFp16Distances = false;
    bool validationPass = false;
};

struct RunMetadata
{
    std::string runId;
    std::string timestamp;
    std::string preset;

    std::string gpuName;
    int computeMajor = 0;
    int computeMinor = 0;
    int multiprocessors = 0;
    std::size_t totalGlobalMemoryBytes = 0;
    int asyncEngineCount = 0;
    bool concurrentKernels = false;

    int cudaRuntimeVersion = 0;
    int cudaDriverVersion = 0;

    std::string gitCommit;
    bool gitDirty = true;

    int warmupRuns = 0;
    int measuredRuns = 0;
    int autotuneTrials = AUTOTUNE_TRIALS;

    unsigned int databaseSeed = DATABASE_SEED;
    unsigned int querySeed = QUERY_SEED;
};

struct CaseExecution
{
    CaseSummary summary;
    std::vector<RawSample> rawSamples;
};

std::string nowTimestamp()
{
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t value =
        std::chrono::system_clock::to_time_t(
            now
        );

    std::tm local{};

#ifdef _WIN32
    localtime_s(
        &local,
        &value
    );
#else
    localtime_r(
        &value,
        &local
    );
#endif

    std::ostringstream stream;

    stream
        << std::put_time(
            &local,
            "%Y-%m-%dT%H:%M:%S"
        );

    return stream.str();
}

std::string makeRunId()
{
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t value =
        std::chrono::system_clock::to_time_t(
            now
        );

    std::tm local{};

#ifdef _WIN32
    localtime_s(
        &local,
        &value
    );
#else
    localtime_r(
        &value,
        &local
    );
#endif

    std::ostringstream stream;

    stream
        << std::put_time(
            &local,
            "%Y%m%d_%H%M%S"
        );

    return stream.str();
}

std::string formatCudaVersion(
    const int version)
{
    if (version <= 0)
    {
        return "unknown";
    }

    const int major =
        version / 1000;

    const int minor =
        (version % 1000) / 10;

    std::ostringstream stream;

    stream
        << major
        << '.'
        << minor;

    return stream.str();
}

std::string jsonEscape(
    const std::string& value)
{
    std::ostringstream stream;

    for (const char character : value)
    {
        switch (character)
        {
            case '\\':
                stream << "\\\\";
                break;

            case '"':
                stream << "\\\"";
                break;

            case '\n':
                stream << "\\n";
                break;

            case '\r':
                stream << "\\r";
                break;

            case '\t':
                stream << "\\t";
                break;

            default:
                stream << character;
                break;
        }
    }

    return stream.str();
}

std::string csvEscape(
    const std::string& value)
{
    const bool needsQuotes =
        value.find(',') !=
            std::string::npos ||
        value.find('"') !=
            std::string::npos ||
        value.find('\n') !=
            std::string::npos ||
        value.find('\r') !=
            std::string::npos;

    if (!needsQuotes)
    {
        return value;
    }

    std::string escaped = "\"";

    for (const char character : value)
    {
        if (character == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += character;
        }
    }

    escaped += '"';

    return escaped;
}

std::string boolText(
    const bool value)
{
    return
        value
            ? "true"
            : "false";
}

BenchmarkPreset presetFromName(
    const std::string_view name)
{
    if (name == "quick")
    {
        return BenchmarkPreset{
            "quick",
            {100000, 500000},
            {128, 129},
            {10},
            {1, 8, 32},
            3,
            15
        };
    }

    if (name == "full")
    {
        return BenchmarkPreset{
            "full",
            {
                100000,
                250000,
                500000,
                1000000
            },
            {128, 129},
            {10, 32},
            {1, 8, 32},
            5,
            30
        };
    }

    throw std::invalid_argument(
        "Preset must be 'quick' or 'full'."
    );
}

std::size_t checkedMultiply(
    const std::size_t left,
    const std::size_t right,
    const char* message)
{
    if (left != 0 &&
        right >
            std::numeric_limits<
                std::size_t
            >::max() /
            left)
    {
        throw std::overflow_error(
            message
        );
    }

    return left * right;
}

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

std::vector<float> prefixQueries(
    const std::vector<float>& queries,
    const int queryCount,
    const int dimension)
{
    const std::size_t count =
        checkedMultiply(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                dimension
            ),
            "Query dimensions overflow size_t."
        );

    if (queries.size() <
        count)
    {
        throw std::invalid_argument(
            "Source query buffer is too small."
        );
    }

    return std::vector<float>(
        queries.begin(),
        queries.begin() +
            static_cast<std::ptrdiff_t>(
                count
            )
    );
}

double percentile(
    const std::vector<double>& sorted,
    const double quantile)
{
    if (sorted.empty())
    {
        throw std::invalid_argument(
            "Cannot calculate percentile from an empty sample."
        );
    }

    if (sorted.size() == 1)
    {
        return sorted.front();
    }

    const double position =
        quantile *
        static_cast<double>(
            sorted.size() - 1
        );

    const std::size_t lower =
        static_cast<std::size_t>(
            std::floor(
                position
            )
        );

    const std::size_t upper =
        static_cast<std::size_t>(
            std::ceil(
                position
            )
        );

    if (lower == upper)
    {
        return sorted[lower];
    }

    const double fraction =
        position -
        static_cast<double>(
            lower
        );

    return
        sorted[lower] +
        fraction *
        (
            sorted[upper] -
            sorted[lower]
        );
}

TimingStats calculateStats(
    std::vector<double> values)
{
    if (values.empty())
    {
        throw std::invalid_argument(
            "Cannot calculate timing statistics from an empty sample."
        );
    }

    std::sort(
        values.begin(),
        values.end()
    );

    TimingStats stats;

    stats.minimum =
        values.front();

    stats.mean =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) /
        static_cast<double>(
            values.size()
        );

    double variance = 0.0;

    for (const double value : values)
    {
        const double delta =
            value -
            stats.mean;

        variance +=
            delta *
            delta;
    }

    variance /=
        static_cast<double>(
            values.size()
        );

    stats.standardDeviation =
        std::sqrt(
            variance
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

bool allDistancesFinite(
    const std::vector<Neighbor>& neighbors)
{
    return
        std::all_of(
            neighbors.begin(),
            neighbors.end(),
            [](const Neighbor& neighbor)
            {
                return
                    std::isfinite(
                        neighbor.distance
                    );
            }
        );
}

double recallAtDepth(
    const std::vector<Neighbor>& expected,
    const std::vector<Neighbor>& actual,
    const int queryCount,
    const int topK,
    const int depth)
{
    if (depth <= 0 ||
        depth > topK)
    {
        throw std::invalid_argument(
            "Recall depth must be between 1 and topK."
        );
    }

    const std::size_t expectedSize =
        checkedMultiply(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                topK
            ),
            "Recall result size overflow."
        );

    if (expected.size() !=
            expectedSize ||
        actual.size() !=
            expectedSize)
    {
        throw std::invalid_argument(
            "Recall result sizes do not match queryCount * topK."
        );
    }

    std::size_t matches = 0;

    for (int queryIndex = 0;
         queryIndex < queryCount;
         ++queryIndex)
    {
        const std::size_t base =
            static_cast<std::size_t>(
                queryIndex
            ) *
            static_cast<std::size_t>(
                topK
            );

        for (int actualRank = 0;
             actualRank < depth;
             ++actualRank)
        {
            const int actualIndex =
                actual[
                    base +
                    static_cast<std::size_t>(
                        actualRank
                    )
                ].index;

            bool found = false;

            for (int expectedRank = 0;
                 expectedRank < depth;
                 ++expectedRank)
            {
                if (expected[
                        base +
                        static_cast<std::size_t>(
                            expectedRank
                        )
                    ].index ==
                    actualIndex)
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                ++matches;
            }
        }
    }

    const std::size_t denominator =
        checkedMultiply(
            static_cast<std::size_t>(
                queryCount
            ),
            static_cast<std::size_t>(
                depth
            ),
            "Recall denominator overflow."
        );

    return
        static_cast<double>(
            matches
        ) /
        static_cast<double>(
            denominator
        );
}

std::pair<double, double>
matchedDistanceErrors(
    const std::vector<Neighbor>& expected,
    const std::vector<Neighbor>& actual,
    const int queryCount,
    const int topK)
{
    double sum = 0.0;
    double maximum = 0.0;
    std::size_t matched = 0;

    for (int queryIndex = 0;
         queryIndex < queryCount;
         ++queryIndex)
    {
        const std::size_t base =
            static_cast<std::size_t>(
                queryIndex
            ) *
            static_cast<std::size_t>(
                topK
            );

        for (int actualRank = 0;
             actualRank < topK;
             ++actualRank)
        {
            const Neighbor& actualNeighbor =
                actual[
                    base +
                    static_cast<std::size_t>(
                        actualRank
                    )
                ];

            for (int expectedRank = 0;
                 expectedRank < topK;
                 ++expectedRank)
            {
                const Neighbor& expectedNeighbor =
                    expected[
                        base +
                        static_cast<std::size_t>(
                            expectedRank
                        )
                    ];

                if (actualNeighbor.index ==
                    expectedNeighbor.index)
                {
                    const double error =
                        std::abs(
                            static_cast<double>(
                                actualNeighbor.distance
                            ) -
                            static_cast<double>(
                                expectedNeighbor.distance
                            )
                        );

                    sum += error;
                    maximum =
                        std::max(
                            maximum,
                            error
                        );
                    ++matched;
                    break;
                }
            }
        }
    }

    if (matched == 0)
    {
        return {
            std::numeric_limits<
                double
            >::infinity(),
            std::numeric_limits<
                double
            >::infinity()
        };
    }

    return {
        sum /
            static_cast<double>(
                matched
            ),
        maximum
    };
}

void appendSample(
    std::vector<RawSample>& raw,
    const std::string& runId,
    const std::string& preset,
    const int vectorCount,
    const int dimension,
    const int topK,
    const int batchSize,
    const int runIndex,
    const std::string& ordering,
    const std::string& path,
    const GpuBatchTopKResult& result)
{
    RawSample sample;

    sample.runId =
        runId;
    sample.preset =
        preset;
    sample.vectorCount =
        vectorCount;
    sample.dimension =
        dimension;
    sample.topK =
        topK;
    sample.batchSize =
        batchSize;
    sample.runIndex =
        runIndex;
    sample.ordering =
        ordering;
    sample.path =
        path;
    sample.queryTileSize =
        result.queryTileSize;
    sample.vectorizedLoadsUsed =
        result.vectorizedLoadsUsed;
    sample.fmaTunedKernelUsed =
        result.fmaTunedKernelUsed;
    sample.mixedPrecisionUsed =
        result.mixedPrecisionUsed;
    sample.half2LoadsUsed =
        result.half2LoadsUsed;
    sample.totalTimeMs =
        result.totalTimeMs;
    sample.fusedKernelTimeMs =
        result.fusedKernelTimeMs;
    sample.mergeKernelTimeMs =
        result.mergeKernelTimeMs;
    sample.queryUploadTimeMs =
        result.queryUploadTimeMs;
    sample.resultDownloadTimeMs =
        result.resultDownloadTimeMs;

    raw.push_back(
        std::move(
            sample
        )
    );
}

CaseExecution benchmarkCase(
    GpuVectorIndex& index,
    const std::vector<float>& sourceQueries,
    const RunMetadata& metadata,
    const int vectorCount,
    const int dimension,
    const int topK,
    const int batchSize)
{
    const std::vector<float> queries =
        prefixQueries(
            sourceQueries,
            batchSize,
            dimension
        );

    const GpuQueryTileTuningResult tuning =
        index.autotuneVectorizedQueryTileSize(
            queries,
            batchSize,
            topK,
            AUTOTUNE_TRIALS
        );

    for (int warmup = 0;
         warmup <
             metadata.warmupRuns;
         ++warmup)
    {
        if ((warmup & 1) == 0)
        {
            (void)
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );

            (void)
                index.searchTopKBatchTiledFp16Auto(
                    queries,
                    batchSize,
                    topK
                );
        }
        else
        {
            (void)
                index.searchTopKBatchTiledFp16Auto(
                    queries,
                    batchSize,
                    topK
                );

            (void)
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );
        }
    }

    std::vector<double> fp32Total;
    std::vector<double> fp16Total;
    std::vector<double> fp32Kernel;
    std::vector<double> fp16Kernel;

    fp32Total.reserve(
        static_cast<std::size_t>(
            metadata.measuredRuns
        )
    );

    fp16Total.reserve(
        static_cast<std::size_t>(
            metadata.measuredRuns
        )
    );

    fp32Kernel.reserve(
        static_cast<std::size_t>(
            metadata.measuredRuns
        )
    );

    fp16Kernel.reserve(
        static_cast<std::size_t>(
            metadata.measuredRuns
        )
    );

    CaseExecution execution;

    execution.rawSamples.reserve(
        static_cast<std::size_t>(
            metadata.measuredRuns
        ) *
        2U
    );

    GpuBatchTopKResult finalFp32;
    GpuBatchTopKResult finalFp16;

    for (int run = 0;
         run <
             metadata.measuredRuns;
         ++run)
    {
        const bool fp32First =
            (run & 1) == 0;

        const std::string ordering =
            fp32First
                ? "FP32_FIRST"
                : "FP16_FIRST";

        GpuBatchTopKResult fp32;
        GpuBatchTopKResult fp16;

        if (fp32First)
        {
            fp32 =
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );

            fp16 =
                index.searchTopKBatchTiledFp16Auto(
                    queries,
                    batchSize,
                    topK
                );
        }
        else
        {
            fp16 =
                index.searchTopKBatchTiledFp16Auto(
                    queries,
                    batchSize,
                    topK
                );

            fp32 =
                index.searchTopKBatchTiledVectorizedFmaAuto(
                    queries,
                    batchSize,
                    topK
                );
        }

        fp32Total.push_back(
            fp32.totalTimeMs
        );

        fp16Total.push_back(
            fp16.totalTimeMs
        );

        fp32Kernel.push_back(
            static_cast<double>(
                fp32.fusedKernelTimeMs
            )
        );

        fp16Kernel.push_back(
            static_cast<double>(
                fp16.fusedKernelTimeMs
            )
        );

        appendSample(
            execution.rawSamples,
            metadata.runId,
            metadata.preset,
            vectorCount,
            dimension,
            topK,
            batchSize,
            run,
            ordering,
            "M12_FP32",
            fp32
        );

        appendSample(
            execution.rawSamples,
            metadata.runId,
            metadata.preset,
            vectorCount,
            dimension,
            topK,
            batchSize,
            run,
            ordering,
            "M14_FP16",
            fp16
        );

        if (run ==
            metadata.measuredRuns - 1)
        {
            finalFp32 =
                std::move(
                    fp32
                );

            finalFp16 =
                std::move(
                    fp16
                );
        }
    }

    CaseSummary& summary =
        execution.summary;

    summary.vectorCount =
        vectorCount;
    summary.dimension =
        dimension;
    summary.topK =
        topK;
    summary.batchSize =
        batchSize;
    summary.queryTileSize =
        tuning.selectedTileSize;
    summary.half2LoadsUsed =
        finalFp16.half2LoadsUsed;

    summary.fp32IndexBytes =
        index.indexBytes();
    summary.fp16IndexBytes =
        index.fp16IndexBytes();

    summary.memoryReductionPercent =
        100.0 *
        (
            1.0 -
            static_cast<double>(
                summary.fp16IndexBytes
            ) /
            static_cast<double>(
                summary.fp32IndexBytes
            )
        );

    summary.fp32IndexUploadTimeMs =
        index.indexUploadTimeMs();

    summary.fp16IndexBuildTimeMs =
        index.fp16IndexBuildTimeMs();

    summary.fp32Total =
        calculateStats(
            fp32Total
        );

    summary.fp16Total =
        calculateStats(
            fp16Total
        );

    summary.fp32Kernel =
        calculateStats(
            fp32Kernel
        );

    summary.fp16Kernel =
        calculateStats(
            fp16Kernel
        );

    summary.fp32Qps =
        static_cast<double>(
            batchSize
        ) *
        1000.0 /
        summary.fp32Total.p50;

    summary.fp16Qps =
        static_cast<double>(
            batchSize
        ) *
        1000.0 /
        summary.fp16Total.p50;

    summary.throughputGain =
        summary.fp16Qps /
        summary.fp32Qps;

    summary.recallAt1 =
        recallAtDepth(
            finalFp32.neighbors,
            finalFp16.neighbors,
            batchSize,
            topK,
            1
        );

    summary.recallAtK =
        recallAtDepth(
            finalFp32.neighbors,
            finalFp16.neighbors,
            batchSize,
            topK,
            topK
        );

    const auto [meanError, maxError] =
        matchedDistanceErrors(
            finalFp32.neighbors,
            finalFp16.neighbors,
            batchSize,
            topK
        );

    summary.meanMatchedDistanceError =
        meanError;

    summary.maxMatchedDistanceError =
        maxError;

    summary.finiteFp16Distances =
        allDistancesFinite(
            finalFp16.neighbors
        );

    summary.validationPass =
        summary.finiteFp16Distances &&
        summary.recallAtK >=
            MIN_RECALL_AT_K;

    return execution;
}

RunMetadata buildMetadata(
    const BenchmarkPreset& preset,
    const cudaDeviceProp& device)
{
    RunMetadata metadata;

    metadata.runId =
        makeRunId();

    metadata.timestamp =
        nowTimestamp();

    metadata.preset =
        preset.name;

    metadata.gpuName =
        device.name;

    metadata.computeMajor =
        device.major;

    metadata.computeMinor =
        device.minor;

    metadata.multiprocessors =
        device.multiProcessorCount;

    metadata.totalGlobalMemoryBytes =
        device.totalGlobalMem;

    metadata.asyncEngineCount =
        device.asyncEngineCount;

    metadata.concurrentKernels =
        device.concurrentKernels != 0;

    CUDA_CHECK(
        cudaRuntimeGetVersion(
            &metadata.cudaRuntimeVersion
        )
    );

    CUDA_CHECK(
        cudaDriverGetVersion(
            &metadata.cudaDriverVersion
        )
    );

    metadata.gitCommit =
        VORTEX_GIT_COMMIT;

    metadata.gitDirty =
        VORTEX_GIT_DIRTY != 0;

    metadata.warmupRuns =
        preset.warmupRuns;

    metadata.measuredRuns =
        preset.measuredRuns;

    return metadata;
}

void writeRawCsvHeader(
    std::ofstream& output)
{
    output
        << "run_id,preset,vector_count,dimension,top_k,batch_size,"
        << "run_index,ordering,path,query_tile_size,"
        << "vectorized_loads,fma_tuned,mixed_precision,half2_loads,"
        << "total_ms,kernel_ms,merge_ms,upload_ms,download_ms\n";
}

void writeRawCsvRow(
    std::ofstream& output,
    const RawSample& sample)
{
    output
        << csvEscape(sample.runId)
        << ','
        << csvEscape(sample.preset)
        << ','
        << sample.vectorCount
        << ','
        << sample.dimension
        << ','
        << sample.topK
        << ','
        << sample.batchSize
        << ','
        << sample.runIndex
        << ','
        << csvEscape(sample.ordering)
        << ','
        << csvEscape(sample.path)
        << ','
        << sample.queryTileSize
        << ','
        << boolText(sample.vectorizedLoadsUsed)
        << ','
        << boolText(sample.fmaTunedKernelUsed)
        << ','
        << boolText(sample.mixedPrecisionUsed)
        << ','
        << boolText(sample.half2LoadsUsed)
        << ','
        << std::fixed
        << std::setprecision(6)
        << sample.totalTimeMs
        << ','
        << sample.fusedKernelTimeMs
        << ','
        << sample.mergeKernelTimeMs
        << ','
        << sample.queryUploadTimeMs
        << ','
        << sample.resultDownloadTimeMs
        << '\n';
}

void writeSummaryCsvHeader(
    std::ofstream& output)
{
    output
        << "run_id,preset,gpu,compute_capability,vector_count,dimension,"
        << "top_k,batch_size,query_tile_size,half2_loads,"
        << "fp32_index_bytes,fp16_index_bytes,memory_reduction_pct,"
        << "fp32_index_upload_ms,fp16_index_build_ms,"
        << "warmup_runs,measured_runs,"
        << "fp32_min_ms,fp32_mean_ms,fp32_stddev_ms,fp32_p50_ms,"
        << "fp32_p95_ms,fp32_p99_ms,fp32_p99_over_p50,fp32_qps,"
        << "fp16_min_ms,fp16_mean_ms,fp16_stddev_ms,fp16_p50_ms,"
        << "fp16_p95_ms,fp16_p99_ms,fp16_p99_over_p50,fp16_qps,"
        << "throughput_gain,recall_at_1_pct,recall_at_k_pct,"
        << "mean_matched_distance_error,max_matched_distance_error,"
        << "finite_fp16_distances,validation_pass\n";
}

void writeSummaryCsvRow(
    std::ofstream& output,
    const RunMetadata& metadata,
    const CaseSummary& summary)
{
    const std::string capability =
        std::to_string(
            metadata.computeMajor
        ) +
        "." +
        std::to_string(
            metadata.computeMinor
        );

    output
        << csvEscape(metadata.runId)
        << ','
        << csvEscape(metadata.preset)
        << ','
        << csvEscape(metadata.gpuName)
        << ','
        << capability
        << ','
        << summary.vectorCount
        << ','
        << summary.dimension
        << ','
        << summary.topK
        << ','
        << summary.batchSize
        << ','
        << summary.queryTileSize
        << ','
        << boolText(summary.half2LoadsUsed)
        << ','
        << summary.fp32IndexBytes
        << ','
        << summary.fp16IndexBytes
        << ','
        << std::fixed
        << std::setprecision(6)
        << summary.memoryReductionPercent
        << ','
        << summary.fp32IndexUploadTimeMs
        << ','
        << summary.fp16IndexBuildTimeMs
        << ','
        << metadata.warmupRuns
        << ','
        << metadata.measuredRuns
        << ','
        << summary.fp32Total.minimum
        << ','
        << summary.fp32Total.mean
        << ','
        << summary.fp32Total.standardDeviation
        << ','
        << summary.fp32Total.p50
        << ','
        << summary.fp32Total.p95
        << ','
        << summary.fp32Total.p99
        << ','
        << (
            summary.fp32Total.p50 > 0.0
                ? summary.fp32Total.p99 /
                    summary.fp32Total.p50
                : 0.0
        )
        << ','
        << summary.fp32Qps
        << ','
        << summary.fp16Total.minimum
        << ','
        << summary.fp16Total.mean
        << ','
        << summary.fp16Total.standardDeviation
        << ','
        << summary.fp16Total.p50
        << ','
        << summary.fp16Total.p95
        << ','
        << summary.fp16Total.p99
        << ','
        << (
            summary.fp16Total.p50 > 0.0
                ? summary.fp16Total.p99 /
                    summary.fp16Total.p50
                : 0.0
        )
        << ','
        << summary.fp16Qps
        << ','
        << summary.throughputGain
        << ','
        << summary.recallAt1 *
            100.0
        << ','
        << summary.recallAtK *
            100.0
        << ','
        << summary.meanMatchedDistanceError
        << ','
        << summary.maxMatchedDistanceError
        << ','
        << boolText(
            summary.finiteFp16Distances
        )
        << ','
        << boolText(
            summary.validationPass
        )
        << '\n';
}

void writeTimingJson(
    std::ofstream& output,
    const TimingStats& stats,
    const int indent)
{
    const std::string spaces(
        static_cast<std::size_t>(
            indent
        ),
        ' '
    );

    output
        << "{\n"
        << spaces
        << "  \"min_ms\": "
        << stats.minimum
        << ",\n"
        << spaces
        << "  \"mean_ms\": "
        << stats.mean
        << ",\n"
        << spaces
        << "  \"stddev_ms\": "
        << stats.standardDeviation
        << ",\n"
        << spaces
        << "  \"p50_ms\": "
        << stats.p50
        << ",\n"
        << spaces
        << "  \"p95_ms\": "
        << stats.p95
        << ",\n"
        << spaces
        << "  \"p99_ms\": "
        << stats.p99
        << ",\n"
        << spaces
        << "  \"p99_over_p50\": "
        << (
            stats.p50 > 0.0
                ? stats.p99 /
                    stats.p50
                : 0.0
        )
        << '\n'
        << spaces
        << '}';
}

void writeJsonReport(
    const std::filesystem::path& path,
    const RunMetadata& metadata,
    const std::vector<CaseSummary>& summaries,
    const bool overallPass)
{
    std::ofstream output(
        path,
        std::ios::out |
        std::ios::trunc
    );

    if (!output)
    {
        throw std::runtime_error(
            "Unable to create JSON report: " +
            path.string()
        );
    }

    output
        << std::fixed
        << std::setprecision(6)
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"milestone\": 15,\n"
        << "  \"run_id\": \""
        << jsonEscape(metadata.runId)
        << "\",\n"
        << "  \"timestamp\": \""
        << jsonEscape(metadata.timestamp)
        << "\",\n"
        << "  \"preset\": \""
        << jsonEscape(metadata.preset)
        << "\",\n"
        << "  \"reference_path\": \"M12_FP32_exact\",\n"
        << "  \"candidate_path\": \"M14_FP16_storage_FP32_accumulation\",\n"
        << "  \"metadata\": {\n"
        << "    \"gpu\": \""
        << jsonEscape(metadata.gpuName)
        << "\",\n"
        << "    \"compute_capability\": \""
        << metadata.computeMajor
        << '.'
        << metadata.computeMinor
        << "\",\n"
        << "    \"multiprocessors\": "
        << metadata.multiprocessors
        << ",\n"
        << "    \"total_global_memory_bytes\": "
        << metadata.totalGlobalMemoryBytes
        << ",\n"
        << "    \"async_copy_engines\": "
        << metadata.asyncEngineCount
        << ",\n"
        << "    \"concurrent_kernels\": "
        << boolText(metadata.concurrentKernels)
        << ",\n"
        << "    \"cuda_runtime_version\": \""
        << formatCudaVersion(
            metadata.cudaRuntimeVersion
        )
        << "\",\n"
        << "    \"cuda_driver_api_version\": \""
        << formatCudaVersion(
            metadata.cudaDriverVersion
        )
        << "\",\n"
        << "    \"git_commit\": \""
        << jsonEscape(metadata.gitCommit)
        << "\",\n"
        << "    \"git_dirty_at_configure\": "
        << boolText(metadata.gitDirty)
        << ",\n"
        << "    \"database_seed\": "
        << metadata.databaseSeed
        << ",\n"
        << "    \"query_seed\": "
        << metadata.querySeed
        << ",\n"
        << "    \"autotune_trials\": "
        << metadata.autotuneTrials
        << ",\n"
        << "    \"warmup_runs\": "
        << metadata.warmupRuns
        << ",\n"
        << "    \"measured_runs\": "
        << metadata.measuredRuns
        << "\n"
        << "  },\n"
        << "  \"overall_pass\": "
        << boolText(overallPass)
        << ",\n"
        << "  \"cases\": [\n";

    for (std::size_t index = 0;
         index < summaries.size();
         ++index)
    {
        const CaseSummary& summary =
            summaries[index];

        output
            << "    {\n"
            << "      \"vector_count\": "
            << summary.vectorCount
            << ",\n"
            << "      \"dimension\": "
            << summary.dimension
            << ",\n"
            << "      \"top_k\": "
            << summary.topK
            << ",\n"
            << "      \"batch_size\": "
            << summary.batchSize
            << ",\n"
            << "      \"query_tile_size\": "
            << summary.queryTileSize
            << ",\n"
            << "      \"half2_loads\": "
            << boolText(
                summary.half2LoadsUsed
            )
            << ",\n"
            << "      \"fp32_index_bytes\": "
            << summary.fp32IndexBytes
            << ",\n"
            << "      \"fp16_index_bytes\": "
            << summary.fp16IndexBytes
            << ",\n"
            << "      \"memory_reduction_pct\": "
            << summary.memoryReductionPercent
            << ",\n"
            << "      \"fp32_index_upload_ms\": "
            << summary.fp32IndexUploadTimeMs
            << ",\n"
            << "      \"fp16_index_build_ms\": "
            << summary.fp16IndexBuildTimeMs
            << ",\n"
            << "      \"fp32_total\": ";

        writeTimingJson(
            output,
            summary.fp32Total,
            6
        );

        output
            << ",\n"
            << "      \"fp16_total\": ";

        writeTimingJson(
            output,
            summary.fp16Total,
            6
        );

        output
            << ",\n"
            << "      \"fp32_kernel\": ";

        writeTimingJson(
            output,
            summary.fp32Kernel,
            6
        );

        output
            << ",\n"
            << "      \"fp16_kernel\": ";

        writeTimingJson(
            output,
            summary.fp16Kernel,
            6
        );

        output
            << ",\n"
            << "      \"fp32_qps\": "
            << summary.fp32Qps
            << ",\n"
            << "      \"fp16_qps\": "
            << summary.fp16Qps
            << ",\n"
            << "      \"throughput_gain\": "
            << summary.throughputGain
            << ",\n"
            << "      \"recall_at_1_pct\": "
            << summary.recallAt1 *
                100.0
            << ",\n"
            << "      \"recall_at_k_pct\": "
            << summary.recallAtK *
                100.0
            << ",\n"
            << "      \"mean_matched_distance_error\": "
            << summary.meanMatchedDistanceError
            << ",\n"
            << "      \"max_matched_distance_error\": "
            << summary.maxMatchedDistanceError
            << ",\n"
            << "      \"finite_fp16_distances\": "
            << boolText(
                summary.finiteFp16Distances
            )
            << ",\n"
            << "      \"validation_pass\": "
            << boolText(
                summary.validationPass
            )
            << "\n"
            << "    }";

        if (index + 1 <
            summaries.size())
        {
            output << ',';
        }

        output << '\n';
    }

    output
        << "  ]\n"
        << "}\n";
}

void printCaseSummary(
    const CaseSummary& summary)
{
    std::cout
        << std::fixed
        << std::setprecision(3)
        << "  FP32 P50="
        << summary.fp32Total.p50
        << " ms"
        << "  P95="
        << summary.fp32Total.p95
        << "  P99="
        << summary.fp32Total.p99
        << "  QPS="
        << summary.fp32Qps
        << '\n'
        << "  FP16 P50="
        << summary.fp16Total.p50
        << " ms"
        << "  P95="
        << summary.fp16Total.p95
        << "  P99="
        << summary.fp16Total.p99
        << "  QPS="
        << summary.fp16Qps
        << '\n'
        << std::setprecision(6)
        << "  Gain="
        << summary.throughputGain
        << "x"
        << "  Recall@1="
        << summary.recallAt1 *
            100.0
        << "%"
        << "  Recall@K="
        << summary.recallAtK *
            100.0
        << "%"
        << "  half2="
        << (
            summary.half2LoadsUsed
                ? "yes"
                : "no"
        )
        << "  validation="
        << (
            summary.validationPass
                ? "PASS"
                : "FAIL"
        )
        << "\n\n";
}

void printUsage()
{
    std::cerr
        << "Usage:\n"
        << "  vortex_benchmark --benchmark-m15 "
        << "<quick|full> [output-directory]\n\n"
        << "Examples:\n"
        << "  vortex_benchmark --benchmark-m15 quick "
        << ".\\benchmark_results\n"
        << "  vortex_benchmark --benchmark-m15 full "
        << ".\\benchmark_results_full\n";
}

} // namespace

int main(
    const int argc,
    char** argv)
{
    try
    {
        if (argc < 3 ||
            argc > 4 ||
            std::string_view(
                argv[1]
            ) !=
                "--benchmark-m15")
        {
            printUsage();
            return EXIT_FAILURE;
        }

        const BenchmarkPreset preset =
            presetFromName(
                argv[2]
            );

        const std::filesystem::path outputDirectory =
            argc == 4
                ? std::filesystem::path(
                    argv[3]
                )
                : std::filesystem::path(
                    "benchmark_results"
                );

        std::filesystem::create_directories(
            outputDirectory
        );

        int deviceCount = 0;

        CUDA_CHECK(
            cudaGetDeviceCount(
                &deviceCount
            )
        );

        if (deviceCount <= 0)
        {
            std::cerr
                << "No CUDA GPU detected.\n";

            return EXIT_FAILURE;
        }

        CUDA_CHECK(
            cudaSetDevice(
                0
            )
        );

        cudaDeviceProp device{};

        CUDA_CHECK(
            cudaGetDeviceProperties(
                &device,
                0
            )
        );

        const RunMetadata metadata =
            buildMetadata(
                preset,
                device
            );

        const std::filesystem::path summaryPath =
            outputDirectory /
            (
                "vortex_m15_summary_" +
                metadata.runId +
                ".csv"
            );

        const std::filesystem::path rawPath =
            outputDirectory /
            (
                "vortex_m15_raw_" +
                metadata.runId +
                ".csv"
            );

        const std::filesystem::path jsonPath =
            outputDirectory /
            (
                "vortex_m15_report_" +
                metadata.runId +
                ".json"
            );

        std::ofstream summaryCsv(
            summaryPath,
            std::ios::out |
            std::ios::trunc
        );

        std::ofstream rawCsv(
            rawPath,
            std::ios::out |
            std::ios::trunc
        );

        if (!summaryCsv ||
            !rawCsv)
        {
            throw std::runtime_error(
                "Unable to create benchmark CSV files."
            );
        }

        writeSummaryCsvHeader(
            summaryCsv
        );

        writeRawCsvHeader(
            rawCsv
        );

        std::cout
            << "============================================\n"
            << " Vortex CUDA M15 Production Benchmark Harness\n"
            << "============================================\n\n"
            << "Run ID:             "
            << metadata.runId
            << '\n'
            << "Preset:             "
            << metadata.preset
            << '\n'
            << "GPU:                "
            << metadata.gpuName
            << '\n'
            << "Compute capability: "
            << metadata.computeMajor
            << '.'
            << metadata.computeMinor
            << '\n'
            << "CUDA runtime:       "
            << formatCudaVersion(
                metadata.cudaRuntimeVersion
            )
            << '\n'
            << "CUDA driver API:    "
            << formatCudaVersion(
                metadata.cudaDriverVersion
            )
            << '\n'
            << "Git commit:         "
            << metadata.gitCommit
            << (
                metadata.gitDirty
                    ? " (dirty at configure)"
                    : ""
            )
            << '\n'
            << "Warmups/case:       "
            << metadata.warmupRuns
            << '\n'
            << "Measured runs/case: "
            << metadata.measuredRuns
            << '\n'
            << "Database seed:      "
            << metadata.databaseSeed
            << '\n'
            << "Query seed:         "
            << metadata.querySeed
            << "\n\n";

        std::vector<CaseSummary> summaries;
        bool overallPass = true;
        int caseNumber = 0;

        const int totalCases =
            static_cast<int>(
                preset.vectorCounts.size() *
                preset.dimensions.size() *
                preset.topKs.size() *
                preset.batchSizes.size()
            );

        for (const int vectorCount :
             preset.vectorCounts)
        {
            for (const int dimension :
                 preset.dimensions)
            {
                const std::size_t databaseElements =
                    checkedMultiply(
                        static_cast<std::size_t>(
                            vectorCount
                        ),
                        static_cast<std::size_t>(
                            dimension
                        ),
                        "Database dimensions overflow size_t."
                    );

                const int maxBatch =
                    *std::max_element(
                        preset.batchSizes.begin(),
                        preset.batchSizes.end()
                    );

                const std::size_t queryElements =
                    checkedMultiply(
                        static_cast<std::size_t>(
                            maxBatch
                        ),
                        static_cast<std::size_t>(
                            dimension
                        ),
                        "Query dimensions overflow size_t."
                    );

                std::cout
                    << "Preparing dataset "
                    << vectorCount
                    << " x "
                    << dimension
                    << "...\n";

                const std::vector<float> database =
                    generateRandomData(
                        databaseElements,
                        DATABASE_SEED
                    );

                const std::vector<float> sourceQueries =
                    generateRandomData(
                        queryElements,
                        QUERY_SEED
                    );

                GpuVectorIndex index(
                    database,
                    vectorCount,
                    dimension
                );

                for (const int topK :
                     preset.topKs)
                {
                    if (topK >
                            vectorCount ||
                        topK >
                            GpuVectorIndex::MAX_TOP_K)
                    {
                        continue;
                    }

                    for (const int batchSize :
                         preset.batchSizes)
                    {
                        ++caseNumber;

                        std::cout
                            << "["
                            << caseNumber
                            << "/"
                            << totalCases
                            << "] vectors="
                            << vectorCount
                            << " dim="
                            << dimension
                            << " K="
                            << topK
                            << " batch="
                            << batchSize
                            << '\n';

                        CaseExecution execution =
                            benchmarkCase(
                                index,
                                sourceQueries,
                                metadata,
                                vectorCount,
                                dimension,
                                topK,
                                batchSize
                            );

                        writeSummaryCsvRow(
                            summaryCsv,
                            metadata,
                            execution.summary
                        );

                        for (const RawSample& sample :
                             execution.rawSamples)
                        {
                            writeRawCsvRow(
                                rawCsv,
                                sample
                            );
                        }

                        summaryCsv.flush();
                        rawCsv.flush();

                        printCaseSummary(
                            execution.summary
                        );

                        overallPass =
                            overallPass &&
                            execution.summary
                                .validationPass;

                        summaries.push_back(
                            std::move(
                                execution.summary
                            )
                        );
                    }
                }

                CUDA_CHECK(
                    cudaDeviceSynchronize()
                );
            }
        }

        writeJsonReport(
            jsonPath,
            metadata,
            summaries,
            overallPass
        );

        std::cout
            << "============================================\n"
            << " M15 Benchmark Complete\n"
            << "============================================\n"
            << "Cases:       "
            << summaries.size()
            << '\n'
            << "Validation:  "
            << (
                overallPass
                    ? "PASS"
                    : "FAIL"
            )
            << '\n'
            << "Summary CSV: "
            << summaryPath.string()
            << '\n'
            << "Raw CSV:     "
            << rawPath.string()
            << '\n'
            << "JSON report: "
            << jsonPath.string()
            << '\n';

        return
            overallPass
                ? EXIT_SUCCESS
                : EXIT_FAILURE;
    }
    catch (
        const std::exception& exception
    )
    {
        std::cerr
            << "Fatal benchmark error: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
