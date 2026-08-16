package com.vortex.server.api.dto;

import com.vortex.cuda.SearchPrecision;

import java.util.List;
import java.util.UUID;

public record BatchSearchResponse(
        UUID indexId,
        SearchPrecision precision,
        int queryCount,
        int topK,
        List<List<NeighborResponse>> results
) {
}
