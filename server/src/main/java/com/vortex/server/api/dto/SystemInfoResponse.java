package com.vortex.server.api.dto;

import com.vortex.cuda.CudaDeviceInfo;

public record SystemInfoResponse(
        String stage,
        String nativeVersion,
        CudaDeviceInfo cudaDevice,
        int activeIndexes,
        long residentVectorBytes,
        int maxActiveIndexes,
        long maxResidentVectorBytes,
        int maxConcurrentSearches
) {
}
