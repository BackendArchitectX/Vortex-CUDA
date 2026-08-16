package com.vortex.server.service;

import com.vortex.cuda.BatchSearchResult;
import com.vortex.cuda.IndexStats;
import com.vortex.cuda.SearchPrecision;
import com.vortex.cuda.SearchResult;
import com.vortex.cuda.VortexIndex;

import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantReadWriteLock;

final class ManagedIndex
        implements AutoCloseable {

    private final UUID id;
    private final Instant createdAt;
    private final VortexIndex index;
    private final long residentVectorBytes;

    private final ReentrantReadWriteLock lifecycleLock =
            new ReentrantReadWriteLock();

    private boolean closed;

    ManagedIndex(
            UUID id,
            float[] database,
            int vectorCount,
            int dimension
    ) {
        this.id = id;
        this.createdAt = Instant.now();

        this.index =
                new VortexIndex(
                        database,
                        vectorCount,
                        dimension
                );

        IndexStats initialStats =
                this.index.stats();

        this.residentVectorBytes =
                Math.addExact(
                        initialStats.fp32IndexBytes(),
                        initialStats.fp16IndexBytes()
                );
    }

    UUID id() {
        return id;
    }

    Instant createdAt() {
        return createdAt;
    }

    int vectorCount() {
        return index.vectorCount();
    }

    int dimension() {
        return index.dimension();
    }

    long residentVectorBytes() {
        return residentVectorBytes;
    }

    IndexStats stats() {
        ReentrantReadWriteLock.ReadLock readLock =
                lifecycleLock.readLock();

        readLock.lock();

        try {
            ensureOpen();

            return index.stats();
        } finally {
            readLock.unlock();
        }
    }

    SearchResult search(
            float[] query,
            int topK,
            SearchPrecision precision
    ) {
        ReentrantReadWriteLock.ReadLock readLock =
                lifecycleLock.readLock();

        readLock.lock();

        try {
            ensureOpen();

            return index.search(
                    query,
                    topK,
                    precision
            );
        } finally {
            readLock.unlock();
        }
    }

    BatchSearchResult searchBatch(
            float[] queries,
            int queryCount,
            int topK,
            SearchPrecision precision
    ) {
        ReentrantReadWriteLock.ReadLock readLock =
                lifecycleLock.readLock();

        readLock.lock();

        try {
            ensureOpen();

            return index.searchBatch(
                    queries,
                    queryCount,
                    topK,
                    precision
            );
        } finally {
            readLock.unlock();
        }
    }

    @Override
    public void close() {
        ReentrantReadWriteLock.WriteLock writeLock =
                lifecycleLock.writeLock();

        writeLock.lock();

        try {
            if (closed) {
                return;
            }

            closed = true;
            index.close();
        } finally {
            writeLock.unlock();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "Vortex index is closed: " +
                            id
            );
        }
    }
}
