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

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/indexes")
public final class VortexIndexController {

    private static final String VECTOR_COUNT_HEADER =
            "X-Vortex-Vector-Count";

    private static final String DIMENSION_HEADER =
            "X-Vortex-Dimension";

    private final VortexIndexService service;

    public VortexIndexController(
            VortexIndexService service
    ) {
        this.service = service;
    }

    @PostMapping(
            consumes =
                    MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<CreateIndexResponse> create(
            @Valid
            @RequestBody
            CreateIndexRequest request
    ) {
        return created(
                service.create(
                        request
                )
        );
    }

    @PostMapping(
            consumes =
                    MediaType.APPLICATION_OCTET_STREAM_VALUE
    )
    public ResponseEntity<CreateIndexResponse> createBinary(
            @RequestHeader(
                    VECTOR_COUNT_HEADER
            )
            int vectorCount,

            @RequestHeader(
                    DIMENSION_HEADER
            )
            int dimension,

            @RequestBody
            byte[] body
    ) {
        return created(
                service.createBinary(
                        vectorCount,
                        dimension,
                        body
                )
        );
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

    private static ResponseEntity<CreateIndexResponse> created(
            CreateIndexResponse response
    ) {
        URI location =
                URI.create(
                        "/api/v1/indexes/" +
                                response.index()
                                        .id()
                );

        return ResponseEntity
                .created(location)
                .body(response);
    }
}
