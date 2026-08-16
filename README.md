# Vortex CUDA

**GPU-accelerated exact vector search with CUDA, Java JNI, and Spring Boot.**

Vortex CUDA is a portfolio-grade vector-search engine built from the CUDA
kernel level upward. The final M18 release exposes persistent GPU indexes
through a Java API and a production-hardened Spring Boot REST service.

## Architecture

```text
Client
  |
  | HTTP / JSON or float32 binary
  v
Spring Boot REST API
  |
  | validation, API key, request IDs,
  | capacity controls, metrics
  v
VortexIndex Java API
  |
  | JNI
  v
vortex_jni.dll
  |
  v
GpuVectorIndex
  |
  | CUDA kernels
  v
NVIDIA GPU
```

## Highlights

- Exact squared-L2 Top-K search.
- Persistent GPU-resident vector index.
- Hierarchical GPU Top-K with fused distance computation.
- Query-tiled batched kernels.
- `float4` vectorized loads and explicit FP32 FMA path.
- Adaptive block-size and query-tile autotuning.
- FP16 database storage with FP32 accumulation.
- Pinned host memory and dual non-blocking CUDA stream pipeline.
- Reproducible benchmark harness with P50/P95/P99 and CSV/JSON output.
- Java JNI API with opaque handles and lifecycle-safe `AutoCloseable` indexes.
- Spring Boot REST API with JSON and little-endian float32 binary ingestion.
- API-key guard, request correlation IDs and request-size checks.
- Active-index/resident-memory limits and a global GPU search concurrency guard.
- Actuator health, metrics and fail-fast CUDA/JNI startup validation.
- Exception-based CUDA failure propagation instead of process-level `exit()`.
- Reproducible Windows x64 release packaging with SHA-256 manifests.

## Validated benchmark highlights

On the development RTX 3050 6GB Laptop GPU:

- `500K x 128`, exact Top-10: approximately **16.7x lower P50 latency**
  than the project's scalar CPU reference after the fused GPU Top-K stage.
- FP16 vector storage reduces logical vector-index storage by **50%** while
  preserving **99.69% Recall@10** on the `500K x 128` validation workload.
- The M15 full reproducibility matrix completed **48/48 validation cases**;
  the lowest observed Recall@K was **99.609375%**.
- FP16 materially improves some single-query cases but is near-neutral or
  slightly slower for some large batches. The project does not claim a
  universal FP16 latency speedup.

See `docs/BENCHMARKS.md` for methodology and caveats.

## Requirements

Validated Windows development environment:

- Windows 11 x64
- NVIDIA CUDA Toolkit 13.3
- Visual Studio 2022 Build Tools
- CMake 3.24+
- JDK 17+
- Maven 3.6.3+
- NVIDIA GPU matching the configured CUDA architecture

The current CMake configuration targets **sm_86**.

## Build

```powershell
cmake `
    -S . `
    -B build `
    -G "Visual Studio 17 2022" `
    -A x64

cmake --build build --config Release --clean-first

mvn --no-transfer-progress clean package
```

Expected:

```text
build/Release/vortex_cuda.exe
build/Release/vortex_benchmark.exe
build/Release/vortex_jni.dll
server/target/vortex-server-1.0.0.jar
```

## Full M18 validation

```powershell
.\BUILD_AND_TEST_M18.ps1
```

The script validates native/JNI builds, Maven tests, Spring Boot startup,
API-key rejection, request-ID propagation, binary index ingestion, FP32 exact
search, FP16 search, batch search, input validation, capacity limits, metrics
and delete/404 behavior.

Expected final line:

```text
M18 PRODUCTION SMOKE TEST: PASS
```

## Run manually

```powershell
$env:VORTEX_API_KEY = "replace-with-a-strong-secret"

java `
    "-Dvortex.jni.path=D:\path\to\vortex_jni.dll" `
    -jar .\server\target\vortex-server-1.0.0.jar
```

Health:

```text
GET /actuator/health
```

When `VORTEX_API_KEY` is configured, application APIs require:

```text
X-Vortex-Api-Key: <secret>
```

## Main endpoints

```text
POST   /api/v1/indexes
GET    /api/v1/indexes
GET    /api/v1/indexes/{id}
POST   /api/v1/indexes/{id}/search
POST   /api/v1/indexes/{id}/search/batch
DELETE /api/v1/indexes/{id}
GET    /api/v1/system

GET    /actuator/health
GET    /actuator/info
GET    /actuator/metrics
```

## Binary ingestion

```text
Content-Type: application/octet-stream
X-Vortex-Vector-Count: <N>
X-Vortex-Dimension: <D>
```

The body is exactly `N * D` little-endian IEEE-754 float32 values in
vector-major order.

See `docs/BINARY_FORMAT.md`.

## Release package

After M18 is committed and the worktree is clean:

```powershell
.\PACKAGE_M18_RELEASE.ps1
```

This creates:

```text
dist/vortex-cuda-1.0.0-windows-x64/
dist/vortex-cuda-1.0.0-windows-x64.zip
```

with binaries, the Spring Boot JAR, runtime configuration, documentation,
build provenance and SHA-256 hashes.

## Scope

Vortex CUDA is an exact-search engine and engineering project, not a drop-in
replacement for a distributed production vector database. Index data is
process-resident; restart persistence, replication, multi-node sharding and
ANN index structures are outside the 1.0.0 scope.
