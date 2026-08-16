package com.vortex.server.api.dto;

import java.util.List;

public record IndexListResponse(
        int count,
        List<IndexStatsResponse> indexes
) {
}
