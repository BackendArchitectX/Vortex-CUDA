package com.vortex.cuda;

import java.nio.file.Path;

final class VortexNative {

    private static final String EXPLICIT_LIBRARY_PROPERTY =
            "vortex.jni.path";

    static {
        String explicitPath =
                System.getProperty(
                        EXPLICIT_LIBRARY_PROPERTY
                );

        if (explicitPath != null &&
                !explicitPath.isBlank()) {

            System.load(
                    Path.of(explicitPath)
                            .toAbsolutePath()
                            .normalize()
                            .toString()
            );
        } else {
            System.loadLibrary(
                    "vortex_jni"
            );
        }
    }

    private VortexNative() {
    }

    static void ensureLoaded() {
        // Trigger class initialization.
    }

    static native String nativeVersion();

    static native String nativeDeviceName();

    static native int nativeComputeCapabilityMajor();

    static native int nativeComputeCapabilityMinor();

    static native long nativeCudaFreeMemoryBytes();

    static native long nativeCudaTotalMemoryBytes();

    static native int nativeCudaRuntimeVersion();

    static native int nativeCudaDriverVersion();

    static native long nativeCreateIndex(
            float[] database,
            int vectorCount,
            int dimension
    );

    static native void nativeDestroyIndex(
            long handle
    );

    static native void nativeSearch(
            long handle,
            float[] queries,
            int queryCount,
            int topK,
            int precisionCode,
            int[] outputIndices,
            float[] outputDistances
    );

    static native long nativeFp32IndexBytes(
            long handle
    );

    static native long nativeFp16IndexBytes(
            long handle
    );

    static native double nativeFp32UploadTimeMs(
            long handle
    );

    static native double nativeFp16BuildTimeMs(
            long handle
    );
}
