package com.vortex.cuda;

public enum SearchPrecision {

    FP32_EXACT(0),

    FP16_STORAGE_FP32_ACCUMULATION(1);

    private final int nativeCode;

    SearchPrecision(
            int nativeCode
    ) {
        this.nativeCode =
                nativeCode;
    }

    int nativeCode() {
        return nativeCode;
    }
}
