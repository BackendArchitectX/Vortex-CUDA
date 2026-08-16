package com.vortex.server.api;

import com.vortex.server.service.IndexCapacityExceededException;
import com.vortex.server.service.IndexNotFoundException;
import com.vortex.server.service.SearchCapacityExceededException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;

import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.net.URI;
import java.time.Instant;
import java.util.stream.Collectors;

@RestControllerAdvice
public final class ApiExceptionHandler {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(
                    ApiExceptionHandler.class
            );

    @ExceptionHandler(IndexNotFoundException.class)
    public ResponseEntity<ProblemDetail> notFound(
            IndexNotFoundException exception
    ) {
        return problem(
                HttpStatus.NOT_FOUND,
                "INDEX_NOT_FOUND",
                exception.getMessage()
        );
    }

    @ExceptionHandler(IndexCapacityExceededException.class)
    public ResponseEntity<ProblemDetail> indexCapacity(
            IndexCapacityExceededException exception
    ) {
        return problem(
                HttpStatus.INSUFFICIENT_STORAGE,
                "INDEX_CAPACITY_EXCEEDED",
                exception.getMessage()
        );
    }

    @ExceptionHandler(SearchCapacityExceededException.class)
    public ResponseEntity<ProblemDetail> searchCapacity(
            SearchCapacityExceededException exception
    ) {
        return problem(
                HttpStatus.TOO_MANY_REQUESTS,
                "SEARCH_CAPACITY_EXCEEDED",
                exception.getMessage()
        );
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ProblemDetail> badRequest(
            IllegalArgumentException exception
    ) {
        return problem(
                HttpStatus.BAD_REQUEST,
                "INVALID_REQUEST",
                exception.getMessage()
        );
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ProblemDetail> validation(
            MethodArgumentNotValidException exception
    ) {
        String detail =
                exception.getBindingResult()
                        .getFieldErrors()
                        .stream()
                        .map(error ->
                                error.getField() +
                                        ": " +
                                        error.getDefaultMessage()
                        )
                        .collect(
                                Collectors.joining(
                                        "; "
                                )
                        );

        return problem(
                HttpStatus.BAD_REQUEST,
                "VALIDATION_FAILED",
                detail
        );
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ProblemDetail> malformedJson(
            HttpMessageNotReadableException exception
    ) {
        return problem(
                HttpStatus.BAD_REQUEST,
                "MALFORMED_JSON",
                "Request JSON could not be parsed."
        );
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<ProblemDetail> nativeFailure(
            IllegalStateException exception
    ) {
        LOGGER.error(
                "Native engine failure",
                exception
        );

        return problem(
                HttpStatus.SERVICE_UNAVAILABLE,
                "NATIVE_ENGINE_UNAVAILABLE",
                "The native CUDA engine is temporarily unavailable."
        );
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ProblemDetail> unexpected(
            Exception exception
    ) {
        LOGGER.error(
                "Unexpected server failure",
                exception
        );

        return problem(
                HttpStatus.INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "Unexpected server error."
        );
    }

    private static ResponseEntity<ProblemDetail> problem(
            HttpStatus status,
            String code,
            String detail
    ) {
        ProblemDetail problem =
                ProblemDetail.forStatusAndDetail(
                        status,
                        detail
                );

        problem.setTitle(
                status.getReasonPhrase()
        );

        problem.setType(
                URI.create(
                        "urn:vortex:error:" +
                                code.toLowerCase()
                )
        );

        problem.setProperty(
                "code",
                code
        );

        problem.setProperty(
                "timestamp",
                Instant.now()
        );

        String requestId =
                MDC.get(
                        "requestId"
                );

        if (requestId != null) {
            problem.setProperty(
                    "requestId",
                    requestId
            );
        }

        return ResponseEntity
                .status(status)
                .body(problem);
    }
}
