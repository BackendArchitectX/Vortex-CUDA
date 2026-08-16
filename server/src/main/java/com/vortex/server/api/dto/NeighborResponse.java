package com.vortex.server.api.dto;

public record NeighborResponse(
        int index,
        float squaredL2Distance
) {
}
