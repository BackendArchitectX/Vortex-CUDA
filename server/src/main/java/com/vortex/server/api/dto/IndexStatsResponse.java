package com.vortex.server.api.dto;

import java.time.Instant;
import java.util.UUID;

public record IndexStatsResponse(
        UUID id,
        Instant createdAt,
        int vectorCount,
        int dimension,
        long fp32IndexBytes,
        long fp16IndexBytes,
        double logicalMemoryReductionPercent,
        double fp32UploadTimeMs,
        double fp16BuildTimeMs
) {
}
