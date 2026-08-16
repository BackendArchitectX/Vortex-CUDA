package com.vortex.server.config;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@ConfigurationProperties(prefix = "vortex.limits")
@Validated
public record VortexLimitsProperties(
        @Min(1)
        @Max(10_000_000)
        int maxVectorCount,

        @Min(1)
        @Max(65_536)
        int maxDimension,

        @Min(1)
        long maxIndexElements
) {
}
