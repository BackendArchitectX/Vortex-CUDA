package com.vortex.server.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "vortex.security")
public record VortexSecurityProperties(
        String apiKey
) {

    public boolean apiKeyEnabled() {
        return apiKey != null &&
                !apiKey.isBlank();
    }
}
