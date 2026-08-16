package com.vortex.server.service;

import com.vortex.server.config.VortexLimitsProperties;

import jakarta.annotation.PreDestroy;

import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@Component
public final class VortexIndexRegistry {

    private final ConcurrentMap<UUID, ManagedIndex> indexes =
            new ConcurrentHashMap<>();

    private final Object mutationMonitor =
            new Object();

    private final VortexLimitsProperties limits;

    private long residentVectorBytes;

    public VortexIndexRegistry(
            VortexLimitsProperties limits
    ) {
        this.limits = limits;
    }

    ManagedIndex create(
            float[] database,
            int vectorCount,
            int dimension
    ) {
        long expectedVectorBytes =
                estimateResidentVectorBytes(
                        vectorCount,
                        dimension
                );

        synchronized (mutationMonitor) {
            if (indexes.size() >=
                    limits.maxActiveIndexes()) {

                throw new IndexCapacityExceededException(
                        "Maximum active index count reached: " +
                                limits.maxActiveIndexes()
                );
            }

            if (wouldExceedResidentLimit(
                    expectedVectorBytes
            )) {
                throw new IndexCapacityExceededException(
                        "Creating this index would exceed the configured " +
                                "resident vector-memory budget of " +
                                limits.maxResidentVectorBytes() +
                                " bytes."
                );
            }

            UUID id =
                    uniqueId();

            ManagedIndex candidate =
                    new ManagedIndex(
                            id,
                            database,
                            vectorCount,
                            dimension
                    );

            long actualBytes =
                    candidate.residentVectorBytes();

            if (wouldExceedResidentLimit(
                    actualBytes
            )) {
                candidate.close();

                throw new IndexCapacityExceededException(
                        "Native index size exceeded the configured " +
                                "resident vector-memory budget."
                );
            }

            indexes.put(
                    id,
                    candidate
            );

            residentVectorBytes =
                    Math.addExact(
                            residentVectorBytes,
                            actualBytes
                    );

            return candidate;
        }
    }

    ManagedIndex require(UUID id) {
        ManagedIndex index =
                indexes.get(id);

        if (index == null) {
            throw new IndexNotFoundException(id);
        }

        return index;
    }

    List<ManagedIndex> snapshot() {
        return indexes.values()
                .stream()
                .sorted(
                        Comparator.comparing(
                                ManagedIndex::createdAt
                        )
                )
                .toList();
    }

    boolean delete(UUID id) {
        synchronized (mutationMonitor) {
            ManagedIndex removed =
                    indexes.remove(id);

            if (removed == null) {
                return false;
            }

            removed.close();

            residentVectorBytes =
                    Math.max(
                            0L,
                            residentVectorBytes -
                                    removed.residentVectorBytes()
                    );

            return true;
        }
    }

    int size() {
        return indexes.size();
    }

    long residentVectorBytes() {
        synchronized (mutationMonitor) {
            return residentVectorBytes;
        }
    }

    @PreDestroy
    void closeAll() {
        synchronized (mutationMonitor) {
            List<ManagedIndex> current =
                    List.copyOf(
                            indexes.values()
                    );

            indexes.clear();

            for (ManagedIndex index : current) {
                index.close();
            }

            residentVectorBytes = 0L;
        }
    }

    private boolean wouldExceedResidentLimit(
            long additionalBytes
    ) {
        if (additionalBytes >
                limits.maxResidentVectorBytes()) {
            return true;
        }

        return residentVectorBytes >
                limits.maxResidentVectorBytes() -
                        additionalBytes;
    }

    private UUID uniqueId() {
        UUID id;

        do {
            id = UUID.randomUUID();
        } while (indexes.containsKey(id));

        return id;
    }

    private static long estimateResidentVectorBytes(
            int vectorCount,
            int dimension
    ) {
        long elements =
                Math.multiplyExact(
                        (long) vectorCount,
                        (long) dimension
                );

        // Current native implementation keeps both FP32 and FP16
        // persistent vector representations to support precision
        // selection per query.
        return Math.multiplyExact(
                elements,
                6L
        );
    }
}
