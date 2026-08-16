package com.vortex.cuda;

public record IndexStats(
        int vectorCount,
        int dimension,
        long fp32IndexBytes,
        long fp16IndexBytes,
        double fp32UploadTimeMs,
        double fp16BuildTimeMs
) {

    public double logicalMemoryReductionPercent() {
        if (fp32IndexBytes <= 0L) {
            return 0.0;
        }

        return 100.0 *
                (
                        1.0 -
                                (
                                        (double) fp16IndexBytes /
                                                (double) fp32IndexBytes
                                )
                );
    }
}
