package com.vortex.server.api;

import com.vortex.server.api.dto.SystemInfoResponse;
import com.vortex.server.service.VortexIndexService;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/system")
public final class VortexSystemController {

    private final VortexIndexService service;

    public VortexSystemController(
            VortexIndexService service
    ) {
        this.service = service;
    }

    @GetMapping
    public SystemInfoResponse systemInfo() {
        return service.systemInfo();
    }
}
