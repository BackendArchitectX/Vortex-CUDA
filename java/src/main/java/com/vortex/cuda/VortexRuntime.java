package com.vortex.cuda;

public final class VortexRuntime {

    private VortexRuntime() {
    }

    public static String nativeVersion() {
        VortexNative.ensureLoaded();

        return VortexNative.nativeVersion();
    }

    public static CudaDeviceInfo deviceInfo() {
        VortexNative.ensureLoaded();

        return new CudaDeviceInfo(
                VortexNative.nativeDeviceName(),
                VortexNative.nativeComputeCapabilityMajor(),
                VortexNative.nativeComputeCapabilityMinor(),
                VortexNative.nativeCudaFreeMemoryBytes(),
                VortexNative.nativeCudaTotalMemoryBytes(),
                VortexNative.nativeCudaRuntimeVersion(),
                VortexNative.nativeCudaDriverVersion()
        );
    }
}
