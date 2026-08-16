package com.vortex.cuda;

import java.lang.ref.Cleaner;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class VortexIndex
        implements AutoCloseable {

    private static final Cleaner CLEANER =
            Cleaner.create();

    private static final int MAX_BATCH_SIZE =
            32;

    private static final int MAX_TOP_K =
            32;

    static {
        VortexNative.ensureLoaded();
    }

    private final int vectorCount;
    private final int dimension;

    private final ReentrantReadWriteLock
            lifecycleLock =
            new ReentrantReadWriteLock();

    private final NativeState nativeState;

    private final Cleaner.Cleanable cleanable;

    private volatile boolean closed;

    public VortexIndex(
            float[] database,
            int vectorCount,
            int dimension
    ) {
        Objects.requireNonNull(
                database,
                "database"
        );

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

        int expected =
                Math.multiplyExact(
                        vectorCount,
                        dimension
                );

        if (database.length !=
                expected) {

            throw new IllegalArgumentException(
                    "database.length must equal vectorCount * dimension."
            );
        }

        long handle =
                VortexNative.nativeCreateIndex(
                        database,
                        vectorCount,
                        dimension
                );

        if (handle == 0L) {
            throw new IllegalStateException(
                    "Native Vortex index creation returned a zero handle."
            );
        }

        this.vectorCount =
                vectorCount;

        this.dimension =
                dimension;

        this.nativeState =
                new NativeState(
                        handle
                );

        this.cleanable =
                CLEANER.register(
                        this,
                        nativeState
                );
    }

    public int vectorCount() {
        return vectorCount;
    }

    public int dimension() {
        return dimension;
    }

    public String nativeVersion() {
        return VortexNative.nativeVersion();
    }

    public SearchResult search(
            float[] query,
            int topK
    ) {
        return search(
                query,
                topK,
                SearchPrecision.FP32_EXACT
        );
    }

    public SearchResult search(
            float[] query,
            int topK,
            SearchPrecision precision
    ) {
        Objects.requireNonNull(
                query,
                "query"
        );

        if (query.length !=
                dimension) {

            throw new IllegalArgumentException(
                    "query.length must equal index dimension."
            );
        }

        BatchSearchResult batch =
                searchBatch(
                        query,
                        1,
                        topK,
                        precision
                );

        return batch.resultForQuery(
                0
        );
    }

    public SearchResult searchFp16(
            float[] query,
            int topK
    ) {
        return search(
                query,
                topK,
                SearchPrecision
                        .FP16_STORAGE_FP32_ACCUMULATION
        );
    }

    public BatchSearchResult searchBatch(
            float[] queries,
            int queryCount,
            int topK
    ) {
        return searchBatch(
                queries,
                queryCount,
                topK,
                SearchPrecision.FP32_EXACT
        );
    }

    public BatchSearchResult searchBatch(
            float[] queries,
            int queryCount,
            int topK,
            SearchPrecision precision
    ) {
        Objects.requireNonNull(
                queries,
                "queries"
        );

        Objects.requireNonNull(
                precision,
                "precision"
        );

        if (queryCount <= 0 ||
                queryCount >
                        MAX_BATCH_SIZE) {

            throw new IllegalArgumentException(
                    "queryCount must be between 1 and 32."
            );
        }

        if (topK <= 0 ||
                topK >
                        MAX_TOP_K ||
                topK >
                        vectorCount) {

            throw new IllegalArgumentException(
                    "topK must be between 1 and min(vectorCount, 32)."
            );
        }

        int expectedQueryElements =
                Math.multiplyExact(
                        queryCount,
                        dimension
                );

        if (queries.length !=
                expectedQueryElements) {

            throw new IllegalArgumentException(
                    "queries.length must equal queryCount * dimension."
            );
        }

        int resultElements =
                Math.multiplyExact(
                        queryCount,
                        topK
                );

        int[] indices =
                new int[
                        resultElements
                ];

        float[] distances =
                new float[
                        resultElements
                ];

        ReentrantReadWriteLock.ReadLock
                readLock =
                lifecycleLock.readLock();

        readLock.lock();

        try {
            ensureOpen();

            VortexNative.nativeSearch(
                    nativeState.requireHandle(),
                    queries,
                    queryCount,
                    topK,
                    precision.nativeCode(),
                    indices,
                    distances
            );
        } finally {
            readLock.unlock();
        }

        return new BatchSearchResult(
                queryCount,
                topK,
                indices,
                distances
        );
    }

    public BatchSearchResult searchBatchFp16(
            float[] queries,
            int queryCount,
            int topK
    ) {
        return searchBatch(
                queries,
                queryCount,
                topK,
                SearchPrecision
                        .FP16_STORAGE_FP32_ACCUMULATION
        );
    }

    public IndexStats stats() {
        ReentrantReadWriteLock.ReadLock
                readLock =
                lifecycleLock.readLock();

        readLock.lock();

        try {
            ensureOpen();

            long handle =
                    nativeState
                            .requireHandle();

            return new IndexStats(
                    vectorCount,
                    dimension,
                    VortexNative
                            .nativeFp32IndexBytes(
                                    handle
                            ),
                    VortexNative
                            .nativeFp16IndexBytes(
                                    handle
                            ),
                    VortexNative
                            .nativeFp32UploadTimeMs(
                                    handle
                            ),
                    VortexNative
                            .nativeFp16BuildTimeMs(
                                    handle
                            )
            );
        } finally {
            readLock.unlock();
        }
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        ReentrantReadWriteLock.WriteLock
                writeLock =
                lifecycleLock.writeLock();

        writeLock.lock();

        try {
            if (closed) {
                return;
            }

            closed = true;

            cleanable.clean();
        } finally {
            writeLock.unlock();
        }
    }

    private void ensureOpen() {
        if (closed ||
                nativeState.handle() ==
                        0L) {

            throw new IllegalStateException(
                    "VortexIndex is closed."
            );
        }
    }

    private static final class NativeState
            implements Runnable {

        private final AtomicLong handle;

        NativeState(
                long handle
        ) {
            this.handle =
                    new AtomicLong(
                            handle
                    );
        }

        long handle() {
            return handle.get();
        }

        long requireHandle() {
            long value =
                    handle.get();

            if (value == 0L) {
                throw new IllegalStateException(
                        "Native Vortex index is closed."
                );
            }

            return value;
        }

        @Override
        public void run() {
            long value =
                    handle.getAndSet(
                            0L
                    );

            if (value != 0L) {
                VortexNative.nativeDestroyIndex(
                        value
                );
            }
        }
    }
}
