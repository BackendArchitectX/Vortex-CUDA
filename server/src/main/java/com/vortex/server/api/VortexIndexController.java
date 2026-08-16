package com.vortex.server.api;

import com.vortex.server.api.dto.BatchSearchRequest;
import com.vortex.server.api.dto.BatchSearchResponse;
import com.vortex.server.api.dto.CreateIndexRequest;
import com.vortex.server.api.dto.CreateIndexResponse;
import com.vortex.server.api.dto.IndexListResponse;
import com.vortex.server.api.dto.IndexStatsResponse;
import com.vortex.server.api.dto.SearchRequest;
import com.vortex.server.api.dto.SearchResponse;
import com.vortex.server.service.VortexIndexService;

import jakarta.validation.Valid;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/indexes")
public final class VortexIndexController {

    private final VortexIndexService service;

    public VortexIndexController(
            VortexIndexService service
    ) {
        this.service = service;
    }

    @PostMapping
    public ResponseEntity<CreateIndexResponse> create(
            @Valid
            @RequestBody
            CreateIndexRequest request
    ) {
        CreateIndexResponse response =
                service.create(request);

        URI location =
                URI.create(
                        "/api/v1/indexes/" +
                                response.index().id()
                );

        return ResponseEntity
                .created(location)
                .body(response);
    }

    @GetMapping
    public IndexListResponse list() {
        return service.list();
    }

    @GetMapping("/{id}")
    public IndexStatsResponse get(
            @PathVariable
            UUID id
    ) {
        return service.get(id);
    }

    @PostMapping("/{id}/search")
    public SearchResponse search(
            @PathVariable
            UUID id,

            @Valid
            @RequestBody
            SearchRequest request
    ) {
        return service.search(
                id,
                request
        );
    }

    @PostMapping("/{id}/search/batch")
    public BatchSearchResponse searchBatch(
            @PathVariable
            UUID id,

            @Valid
            @RequestBody
            BatchSearchRequest request
    ) {
        return service.searchBatch(
                id,
                request
        );
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(
            @PathVariable
            UUID id
    ) {
        service.delete(id);

        return ResponseEntity
                .noContent()
                .build();
    }
}
