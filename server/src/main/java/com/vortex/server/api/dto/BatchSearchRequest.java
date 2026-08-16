package com.vortex.server.api.dto;

import com.vortex.cuda.SearchPrecision;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public record BatchSearchRequest(
        @NotNull
        float[] queries,

        @Min(1)
        @Max(32)
        int queryCount,

        @Min(1)
        @Max(32)
        int topK,

        SearchPrecision precision
) {
}
