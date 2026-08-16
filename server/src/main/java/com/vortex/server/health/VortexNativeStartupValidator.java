package com.vortex.server.health;

import com.vortex.cuda.CudaDeviceInfo;
import com.vortex.cuda.VortexRuntime;
import com.vortex.server.config.VortexNativeProperties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public final class VortexNativeStartupValidator
        implements ApplicationRunner {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(
                    VortexNativeStartupValidator.class
            );

    private final VortexNativeProperties properties;

    public VortexNativeStartupValidator(
            VortexNativeProperties properties
    ) {
        this.properties = properties;
    }

    @Override
    public void run(
            ApplicationArguments arguments
    ) {
        try {
            String version =
                    VortexRuntime.nativeVersion();

            CudaDeviceInfo device =
                    VortexRuntime.deviceInfo();

            LOGGER.info(
                    "nativeEngineReady version=\"{}\" device=\"{}\" computeCapability={} freeMemoryBytes={}",
                    version,
                    device.name(),
                    device.computeCapability(),
                    device.freeMemoryBytes()
            );
        } catch (LinkageError | RuntimeException exception) {
            if (properties.failFast()) {
                throw new IllegalStateException(
                        "CUDA/JNI startup validation failed.",
                        exception
                );
            }

            LOGGER.warn(
                    "CUDA/JNI startup validation failed but fail-fast is disabled.",
                    exception
            );
        }
    }
}
