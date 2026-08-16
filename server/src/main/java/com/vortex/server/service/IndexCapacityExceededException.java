package com.vortex.server.service;

public final class IndexCapacityExceededException
        extends RuntimeException {

    public IndexCapacityExceededException(
            String message
    ) {
        super(message);
    }
}
