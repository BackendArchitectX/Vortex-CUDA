package com.vortex.server.health;

import com.vortex.cuda.CudaDeviceInfo;
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
            CudaDeviceInfo device =
                    VortexRuntime.deviceInfo();

            return Health.up()
                    .withDetail(
                            "nativeVersion",
                            VortexRuntime.nativeVersion()
                    )
                    .withDetail(
                            "engine",
                            "CUDA JNI"
                    )
                    .withDetail(
                            "device",
                            device.name()
                    )
                    .withDetail(
                            "computeCapability",
                            device.computeCapability()
                    )
                    .withDetail(
                            "freeMemoryBytes",
                            device.freeMemoryBytes()
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
