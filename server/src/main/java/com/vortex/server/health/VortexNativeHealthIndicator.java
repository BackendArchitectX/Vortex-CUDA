package com.vortex.server.health;

import com.vortex.cuda.VortexRuntime;

import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
import org.springframework.stereotype.Component;

@Component
public final class VortexNativeHealthIndicator
        implements HealthIndicator {

    @Override
    public Health health() {
        try {
            String version =
                    VortexRuntime.nativeVersion();

            return Health.up()
                    .withDetail(
                            "nativeVersion",
                            version
                    )
                    .withDetail(
                            "engine",
                            "CUDA JNI"
                    )
                    .build();
        } catch (LinkageError | RuntimeException exception) {
            return Health.down(exception)
                    .withDetail(
                            "engine",
                            "CUDA JNI"
                    )
                    .build();
        }
    }
}
