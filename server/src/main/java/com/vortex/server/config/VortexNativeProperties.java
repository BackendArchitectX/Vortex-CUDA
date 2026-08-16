package com.vortex.server.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "vortex.native")
public record VortexNativeProperties(
        boolean failFast
) {
}
