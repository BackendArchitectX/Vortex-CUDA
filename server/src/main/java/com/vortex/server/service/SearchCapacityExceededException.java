package com.vortex.server.service;

public final class SearchCapacityExceededException
        extends RuntimeException {

    public SearchCapacityExceededException(
            String message
    ) {
        super(message);
    }
}
