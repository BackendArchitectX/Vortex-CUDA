package com.vortex.cuda;

public record CudaDeviceInfo(
        String name,
        int computeCapabilityMajor,
        int computeCapabilityMinor,
        long freeMemoryBytes,
        long totalMemoryBytes,
        int runtimeVersion,
        int driverVersion
) {

    public String computeCapability() {
        return computeCapabilityMajor +
                "." +
                computeCapabilityMinor;
    }

    public String runtimeVersionString() {
        return formatCudaVersion(
                runtimeVersion
        );
    }

    public String driverVersionString() {
        return formatCudaVersion(
                driverVersion
        );
    }

    private static String formatCudaVersion(
            int encodedVersion
    ) {
        if (encodedVersion <= 0) {
            return "unknown";
        }

        int major =
                encodedVersion /
                        1000;

        int minor =
                (
                        encodedVersion %
                                1000
                ) /
                        10;

        return major +
                "." +
                minor;
    }
}
