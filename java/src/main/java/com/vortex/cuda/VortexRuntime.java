package com.vortex.cuda;

public final class VortexRuntime {

    private VortexRuntime() {
    }

    public static String nativeVersion() {
        VortexNative.ensureLoaded();
        return VortexNative.nativeVersion();
    }
}
