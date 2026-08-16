package com.vortex.server.service;

import java.util.UUID;

public final class IndexNotFoundException
        extends RuntimeException {

    public IndexNotFoundException(UUID id) {
        super("Vortex index not found: " + id);
    }
}
