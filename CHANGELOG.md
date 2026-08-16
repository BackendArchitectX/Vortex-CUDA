# Changelog

## 1.0.0 — M18

Production hardening and final release.

- Replaced process-fatal CUDA checks with exception-based CUDA failures.
- Added native CUDA device/runtime diagnostics exposed to Java.
- Added binary little-endian float32 index ingestion.
- Added optional API-key protection for `/api/**`.
- Added `X-Request-ID` correlation and request logging.
- Added request-size and finite float validation.
- Added active-index and resident-vector-memory capacity limits.
- Added global GPU search concurrency permits.
- Added fail-fast CUDA/JNI startup validation.
- Hardened ProblemDetail error responses.
- Added binary codec unit tests.
- Added release packaging and SHA-256 manifests.
- Added final architecture, API, benchmark and security documentation.

## 0.4.0 — M17

- Spring Boot REST backend.
- UUID index registry.
- Single and batch search endpoints.
- FP32/FP16 REST precision selection.
- Actuator health and custom metrics.
- Graceful index cleanup.

## 0.3.0 — M16

- JNI shared library.
- Java `VortexIndex` API.
- Opaque native handle registry.
- Java/native lifecycle and correctness smoke tests.

## Earlier milestones

M0-M15 progressively implemented and profiled the CUDA search engine:
persistent GPU indexes, warp reductions, adaptive launch configuration,
hierarchical/fused GPU Top-K, batched/query-tiled execution, `float4` loads,
FMA tuning, pinned memory/streams, FP16 storage and the reproducible benchmark
harness.
