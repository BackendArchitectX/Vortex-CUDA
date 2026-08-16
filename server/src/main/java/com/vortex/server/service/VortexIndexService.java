package com.vortex.server.service;

import com.vortex.cuda.BatchSearchResult;
import com.vortex.cuda.CudaDeviceInfo;
import com.vortex.cuda.IndexStats;
import com.vortex.cuda.Neighbor;
import com.vortex.cuda.SearchPrecision;
import com.vortex.cuda.SearchResult;
import com.vortex.cuda.VortexRuntime;

import com.vortex.server.api.dto.BatchSearchRequest;
import com.vortex.server.api.dto.BatchSearchResponse;
import com.vortex.server.api.dto.CreateIndexRequest;
import com.vortex.server.api.dto.CreateIndexResponse;
import com.vortex.server.api.dto.IndexListResponse;
import com.vortex.server.api.dto.IndexStatsResponse;
import com.vortex.server.api.dto.NeighborResponse;
import com.vortex.server.api.dto.SearchRequest;
import com.vortex.server.api.dto.SearchResponse;
import com.vortex.server.api.dto.SystemInfoResponse;
import com.vortex.server.config.VortexLimitsProperties;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

@Service
public final class VortexIndexService {

    private static final SearchPrecision DEFAULT_PRECISION =
            SearchPrecision.FP32_EXACT;

    private final VortexIndexRegistry indexRegistry;
    private final VortexLimitsProperties limits;
    private final MeterRegistry meterRegistry;

    private final Semaphore searchPermits;

    private final Counter jsonCreatedCounter;
    private final Counter binaryCreatedCounter;
    private final Counter deletedCounter;
    private final Counter singleSearchCounter;
    private final Counter batchSearchCounter;
    private final Counter rejectedSearchCounter;

    public VortexIndexService(
            VortexIndexRegistry indexRegistry,
            VortexLimitsProperties limits,
            MeterRegistry meterRegistry
    ) {
        this.indexRegistry = indexRegistry;
        this.limits = limits;
        this.meterRegistry = meterRegistry;

        this.searchPermits =
                new Semaphore(
                        limits.maxConcurrentSearches(),
                        true
                );

        this.jsonCreatedCounter =
                Counter.builder(
                                "vortex.index.created"
                        )
                        .tag(
                                "ingest",
                                "json"
                        )
                        .description(
                                "Number of native Vortex indexes created"
                        )
                        .register(meterRegistry);

        this.binaryCreatedCounter =
                Counter.builder(
                                "vortex.index.created"
                        )
                        .tag(
                                "ingest",
                                "binary"
                        )
                        .description(
                                "Number of native Vortex indexes created through binary ingestion"
                        )
                        .register(meterRegistry);

        this.deletedCounter =
                Counter.builder(
                                "vortex.index.deleted"
                        )
                        .description(
                                "Number of native Vortex indexes deleted"
                        )
                        .register(meterRegistry);

        this.singleSearchCounter =
                Counter.builder(
                                "vortex.search.requests"
                        )
                        .tag(
                                "mode",
                                "single"
                        )
                        .register(meterRegistry);

        this.batchSearchCounter =
                Counter.builder(
                                "vortex.search.requests"
                        )
                        .tag(
                                "mode",
                                "batch"
                        )
                        .register(meterRegistry);

        this.rejectedSearchCounter =
                Counter.builder(
                                "vortex.search.rejected"
                        )
                        .description(
                                "Searches rejected because GPU concurrency capacity was unavailable"
                        )
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

        Gauge.builder(
                        "vortex.index.resident.vector.bytes",
                        indexRegistry,
                        registry ->
                                (double)
                                        registry.residentVectorBytes()
                )
                .description(
                        "Logical persistent FP32 plus FP16 vector storage currently resident on the GPU"
                )
                .register(meterRegistry);

        Gauge.builder(
                        "vortex.search.permits.available",
                        searchPermits,
                        semaphore ->
                                (double)
                                        semaphore.availablePermits()
                )
                .description(
                        "Available global GPU search permits"
                )
                .register(meterRegistry);
    }

    public CreateIndexResponse create(
            CreateIndexRequest request
    ) {
        return createInternal(
                request.vectorCount(),
                request.dimension(),
                request.vectors(),
                false
        );
    }

    public CreateIndexResponse createBinary(
            int vectorCount,
            int dimension,
            byte[] body
    ) {
        float[] vectors =
                BinaryVectorCodec
                        .decodeLittleEndianFloat32(
                                body,
                                vectorCount,
                                dimension
                        );

        return createInternal(
                vectorCount,
                dimension,
                vectors,
                true
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

    public SystemInfoResponse systemInfo() {
        CudaDeviceInfo deviceInfo =
                VortexRuntime.deviceInfo();

        return new SystemInfoResponse(
                "M18",
                VortexRuntime.nativeVersion(),
                deviceInfo,
                indexRegistry.size(),
                indexRegistry.residentVectorBytes(),
                limits.maxActiveIndexes(),
                limits.maxResidentVectorBytes(),
                limits.maxConcurrentSearches()
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

        SearchResult result =
                executeWithPermit(
                        "single",
                        precision,
                        () ->
                                managedIndex.search(
                                        request.query(),
                                        request.topK(),
                                        precision
                                )
                );

        singleSearchCounter.increment();

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

        BatchSearchResult result =
                executeWithPermit(
                        "batch",
                        precision,
                        () ->
                                managedIndex.searchBatch(
                                        request.queries(),
                                        request.queryCount(),
                                        request.topK(),
                                        precision
                                )
                );

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

    private CreateIndexResponse createInternal(
            int vectorCount,
            int dimension,
            float[] vectors,
            boolean binaryIngest
    ) {
        validateCreateRequest(
                vectorCount,
                dimension,
                vectors
        );

        ManagedIndex managedIndex =
                indexRegistry.create(
                        vectors,
                        vectorCount,
                        dimension
                );

        if (binaryIngest) {
            binaryCreatedCounter.increment();
        } else {
            jsonCreatedCounter.increment();
        }

        return new CreateIndexResponse(
                toStatsResponse(
                        managedIndex
                )
        );
    }

    private void validateCreateRequest(
            int vectorCount,
            int dimension,
            float[] vectors
    ) {
        if (vectorCount <= 0) {
            throw new IllegalArgumentException(
                    "vectorCount must be positive."
            );
        }

        if (dimension <= 0) {
            throw new IllegalArgumentException(
                    "dimension must be positive."
            );
        }

        if (vectors == null) {
            throw new IllegalArgumentException(
                    "vectors must not be null."
            );
        }

        if (vectorCount >
                limits.maxVectorCount()) {

            throw new IllegalArgumentException(
                    "vectorCount exceeds configured maximum of " +
                            limits.maxVectorCount()
            );
        }

        if (dimension >
                limits.maxDimension()) {

            throw new IllegalArgumentException(
                    "dimension exceeds configured maximum of " +
                            limits.maxDimension()
            );
        }

        long expectedElements =
                Math.multiplyExact(
                        (long) vectorCount,
                        (long) dimension
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

        if (vectors.length !=
                expectedElements) {

            throw new IllegalArgumentException(
                    "vectors.length must equal vectorCount * dimension."
            );
        }

        validateFinite(
                vectors,
                "vectors"
        );
    }

    private void validateSingleSearch(
            ManagedIndex index,
            SearchRequest request
    ) {
        if (request.query() == null) {
            throw new IllegalArgumentException(
                    "query must not be null."
            );
        }

        if (request.query().length !=
                index.dimension()) {

            throw new IllegalArgumentException(
                    "query.length must equal index dimension " +
                            index.dimension()
            );
        }

        validateFinite(
                request.query(),
                "query"
        );

        validateTopK(
                request.topK(),
                index.vectorCount()
        );
    }

    private void validateBatchSearch(
            ManagedIndex index,
            BatchSearchRequest request
    ) {
        if (request.queries() == null) {
            throw new IllegalArgumentException(
                    "queries must not be null."
            );
        }

        if (request.queryCount() <= 0 ||
                request.queryCount() > 32) {

            throw new IllegalArgumentException(
                    "queryCount must be between 1 and 32."
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

        validateFinite(
                request.queries(),
                "queries"
        );

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

    private static void validateFinite(
            float[] values,
            String name
    ) {
        for (int index = 0;
             index < values.length;
             index++) {

            if (!Float.isFinite(
                    values[index]
            )) {
                throw new IllegalArgumentException(
                        name +
                                " contains a non-finite value at index " +
                                index +
                                "."
                );
            }
        }
    }

    private <T> T executeWithPermit(
            String mode,
            SearchPrecision precision,
            Supplier<T> operation
    ) {
        boolean acquired = false;

        try {
            acquired =
                    searchPermits.tryAcquire(
                            limits.searchAcquireTimeoutMs(),
                            TimeUnit.MILLISECONDS
                    );

            if (!acquired) {
                rejectedSearchCounter.increment();

                throw new SearchCapacityExceededException(
                        "GPU search concurrency capacity is currently exhausted."
                );
            }

            long start =
                    System.nanoTime();

            try {
                return operation.get();
            } finally {
                recordLatency(
                        mode,
                        precision,
                        start
                );
            }
        } catch (InterruptedException exception) {
            Thread.currentThread()
                    .interrupt();

            rejectedSearchCounter.increment();

            throw new SearchCapacityExceededException(
                    "Interrupted while waiting for GPU search capacity."
            );
        } finally {
            if (acquired) {
                searchPermits.release();
            }
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
                .tag(
                        "mode",
                        mode
                )
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
