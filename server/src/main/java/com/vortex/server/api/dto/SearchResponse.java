package com.vortex.server.api.dto;

import com.vortex.cuda.SearchPrecision;

import java.util.List;
import java.util.UUID;

public record SearchResponse(
        UUID indexId,
        SearchPrecision precision,
        int topK,
        List<NeighborResponse> neighbors
) {
}
