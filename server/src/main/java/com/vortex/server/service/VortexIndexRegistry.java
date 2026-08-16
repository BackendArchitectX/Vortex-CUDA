package com.vortex.server.service;

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

    ManagedIndex create(
            float[] database,
            int vectorCount,
            int dimension
    ) {
        while (true) {
            UUID id = UUID.randomUUID();

            ManagedIndex candidate =
                    new ManagedIndex(
                            id,
                            database,
                            vectorCount,
                            dimension
                    );

            ManagedIndex previous =
                    indexes.putIfAbsent(
                            id,
                            candidate
                    );

            if (previous == null) {
                return candidate;
            }

            candidate.close();
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
        ManagedIndex removed =
                indexes.remove(id);

        if (removed == null) {
            return false;
        }

        removed.close();
        return true;
    }

    int size() {
        return indexes.size();
    }

    @PreDestroy
    void closeAll() {
        List<ManagedIndex> current =
                List.copyOf(
                        indexes.values()
                );

        indexes.clear();

        for (ManagedIndex index : current) {
            index.close();
        }
    }
}
