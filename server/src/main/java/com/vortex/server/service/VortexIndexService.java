package com.vortex.server.service;

import com.vortex.cuda.BatchSearchResult;
import com.vortex.cuda.IndexStats;
import com.vortex.cuda.Neighbor;
import com.vortex.cuda.SearchPrecision;
import com.vortex.cuda.SearchResult;
import com.vortex.server.api.dto.BatchSearchRequest;
import com.vortex.server.api.dto.BatchSearchResponse;
import com.vortex.server.api.dto.CreateIndexRequest;
import com.vortex.server.api.dto.CreateIndexResponse;
import com.vortex.server.api.dto.IndexListResponse;
import com.vortex.server.api.dto.IndexStatsResponse;
import com.vortex.server.api.dto.NeighborResponse;
import com.vortex.server.api.dto.SearchRequest;
import com.vortex.server.api.dto.SearchResponse;
import com.vortex.server.config.VortexLimitsProperties;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Service
public final class VortexIndexService {

    private static final SearchPrecision DEFAULT_PRECISION =
            SearchPrecision.FP32_EXACT;

    private final VortexIndexRegistry indexRegistry;
    private final VortexLimitsProperties limits;
    private final MeterRegistry meterRegistry;

    private final Counter createdCounter;
    private final Counter deletedCounter;
    private final Counter searchCounter;
    private final Counter batchSearchCounter;

    public VortexIndexService(
            VortexIndexRegistry indexRegistry,
            VortexLimitsProperties limits,
            MeterRegistry meterRegistry
    ) {
        this.indexRegistry = indexRegistry;
        this.limits = limits;
        this.meterRegistry = meterRegistry;

        this.createdCounter = Counter.builder(
                        "vortex.index.created"
                )
                .description(
                        "Number of native Vortex indexes created"
                )
                .register(meterRegistry);

        this.deletedCounter = Counter.builder(
                        "vortex.index.deleted"
                )
                .description(
                        "Number of native Vortex indexes deleted"
                )
                .register(meterRegistry);

        this.searchCounter = Counter.builder(
                        "vortex.search.requests"
                )
                .tag("mode", "single")
                .register(meterRegistry);

        this.batchSearchCounter = Counter.builder(
                        "vortex.search.requests"
                )
                .tag("mode", "batch")
                .register(meterRegistry);

        Gauge.builder(
                        "vortex.index.active",
                        indexRegistry,
                        VortexIndexRegistry::size
                )
                .description(
                        "Number of active native Vortex indexes"
                )
                .register(meterRegistry);
    }

    public CreateIndexResponse create(
            CreateIndexRequest request
    ) {
        validateCreateRequest(request);

        ManagedIndex managedIndex =
                indexRegistry.create(
                        request.vectors(),
                        request.vectorCount(),
                        request.dimension()
                );

        createdCounter.increment();

        return new CreateIndexResponse(
                toStatsResponse(
                        managedIndex
                )
        );
    }

    public IndexListResponse list() {
        List<IndexStatsResponse> indexes =
                indexRegistry.snapshot()
                        .stream()
                        .map(this::toStatsResponse)
                        .toList();

        return new IndexListResponse(
                indexes.size(),
                indexes
        );
    }

    public IndexStatsResponse get(UUID id) {
        return toStatsResponse(
                indexRegistry.require(id)
        );
    }

    public SearchResponse search(
            UUID id,
            SearchRequest request
    ) {
        ManagedIndex managedIndex =
                indexRegistry.require(id);

        validateSingleSearch(
                managedIndex,
                request
        );

        SearchPrecision precision =
                precisionOrDefault(
                        request.precision()
                );

        long start = System.nanoTime();

        SearchResult result;

        try {
            result = managedIndex.search(
                    request.query(),
                    request.topK(),
                    precision
            );
        } finally {
            recordLatency(
                    "single",
                    precision,
                    start
            );
        }

        searchCounter.increment();

        return new SearchResponse(
                id,
                precision,
                request.topK(),
                toNeighbors(result)
        );
    }

    public BatchSearchResponse searchBatch(
            UUID id,
            BatchSearchRequest request
    ) {
        ManagedIndex managedIndex =
                indexRegistry.require(id);

        validateBatchSearch(
                managedIndex,
                request
        );

        SearchPrecision precision =
                precisionOrDefault(
                        request.precision()
                );

        long start = System.nanoTime();

        BatchSearchResult result;

        try {
            result = managedIndex.searchBatch(
                    request.queries(),
                    request.queryCount(),
                    request.topK(),
                    precision
            );
        } finally {
            recordLatency(
                    "batch",
                    precision,
                    start
            );
        }

        batchSearchCounter.increment();

        List<List<NeighborResponse>> responses =
                new ArrayList<>(
                        result.queryCount()
                );

        for (int queryIndex = 0;
             queryIndex < result.queryCount();
             queryIndex++) {

            responses.add(
                    toNeighbors(
                            result.resultForQuery(
                                    queryIndex
                            )
                    )
            );
        }

        return new BatchSearchResponse(
                id,
                precision,
                result.queryCount(),
                result.topK(),
                List.copyOf(responses)
        );
    }

    public void delete(UUID id) {
        if (!indexRegistry.delete(id)) {
            throw new IndexNotFoundException(id);
        }

        deletedCounter.increment();
    }

    private void validateCreateRequest(
            CreateIndexRequest request
    ) {
        if (request.vectorCount() >
                limits.maxVectorCount()) {

            throw new IllegalArgumentException(
                    "vectorCount exceeds configured maximum of " +
                            limits.maxVectorCount()
            );
        }

        if (request.dimension() >
                limits.maxDimension()) {

            throw new IllegalArgumentException(
                    "dimension exceeds configured maximum of " +
                            limits.maxDimension()
            );
        }

        long expectedElements =
                Math.multiplyExact(
                        (long) request.vectorCount(),
                        (long) request.dimension()
                );

        if (expectedElements >
                limits.maxIndexElements()) {

            throw new IllegalArgumentException(
                    "index payload contains " +
                            expectedElements +
                            " elements; configured maximum is " +
                            limits.maxIndexElements()
            );
        }

        if (request.vectors().length !=
                expectedElements) {

            throw new IllegalArgumentException(
                    "vectors.length must equal vectorCount * dimension."
            );
        }
    }

    private void validateSingleSearch(
            ManagedIndex index,
            SearchRequest request
    ) {
        if (request.query().length !=
                index.dimension()) {

            throw new IllegalArgumentException(
                    "query.length must equal index dimension " +
                            index.dimension()
            );
        }

        validateTopK(
                request.topK(),
                index.vectorCount()
        );
    }

    private void validateBatchSearch(
            ManagedIndex index,
            BatchSearchRequest request
    ) {
        if (request.queryCount() > 32) {
            throw new IllegalArgumentException(
                    "queryCount must not exceed 32."
            );
        }

        int expectedElements =
                Math.multiplyExact(
                        request.queryCount(),
                        index.dimension()
                );

        if (request.queries().length !=
                expectedElements) {

            throw new IllegalArgumentException(
                    "queries.length must equal queryCount * index dimension."
            );
        }

        validateTopK(
                request.topK(),
                index.vectorCount()
        );
    }

    private static void validateTopK(
            int topK,
            int vectorCount
    ) {
        if (topK <= 0 ||
                topK > 32 ||
                topK > vectorCount) {

            throw new IllegalArgumentException(
                    "topK must be between 1 and min(vectorCount, 32)."
            );
        }
    }

    private static SearchPrecision precisionOrDefault(
            SearchPrecision precision
    ) {
        return precision == null
                ? DEFAULT_PRECISION
                : precision;
    }

    private IndexStatsResponse toStatsResponse(
            ManagedIndex index
    ) {
        IndexStats stats =
                index.stats();

        return new IndexStatsResponse(
                index.id(),
                index.createdAt(),
                stats.vectorCount(),
                stats.dimension(),
                stats.fp32IndexBytes(),
                stats.fp16IndexBytes(),
                stats.logicalMemoryReductionPercent(),
                stats.fp32UploadTimeMs(),
                stats.fp16BuildTimeMs()
        );
    }

    private static List<NeighborResponse> toNeighbors(
            SearchResult result
    ) {
        List<NeighborResponse> neighbors =
                new ArrayList<>(
                        result.size()
                );

        for (int rank = 0;
             rank < result.size();
             rank++) {

            Neighbor neighbor =
                    result.neighbor(rank);

            neighbors.add(
                    new NeighborResponse(
                            neighbor.index(),
                            neighbor.squaredL2Distance()
                    )
            );
        }

        return List.copyOf(neighbors);
    }

    private void recordLatency(
            String mode,
            SearchPrecision precision,
            long startNanos
    ) {
        long elapsed =
                System.nanoTime() -
                        startNanos;

        Timer.builder(
                        "vortex.search.latency"
                )
                .description(
                        "End-to-end Java service search latency"
                )
                .tag("mode", mode)
                .tag(
                        "precision",
                        precision.name()
                )
                .register(meterRegistry)
                .record(
                        elapsed,
                        TimeUnit.NANOSECONDS
                );
    }
}
