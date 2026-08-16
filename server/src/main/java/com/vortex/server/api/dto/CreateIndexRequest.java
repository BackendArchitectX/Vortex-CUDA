package com.vortex.server.api.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;

public record CreateIndexRequest(
        @Positive
        int vectorCount,

        @Positive
        int dimension,

        @NotNull
        float[] vectors
) {
}
