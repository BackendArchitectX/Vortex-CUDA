# Vortex CUDA

> **GPU-accelerated exact vector search engine built from scratch with CUDA, C++20, Java JNI, and Spring Boot.**

[![Release](https://img.shields.io/github/v/release/BackendArchitectX/Vortex-CUDA?label=release)](../../releases/tag/v1.0.0)
![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900?logo=nvidia&logoColor=white)
![Java](https://img.shields.io/badge/Java-17-ED8B00?logo=openjdk&logoColor=white)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-4.1-6DB33F?logo=springboot&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)

Vortex CUDA is an end-to-end vector-search system that starts at the CUDA kernel level and exposes persistent GPU indexes through a Java API and a hardened Spring Boot REST service.

## Performance at a glance

Validated on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** using exact squared-L2 search:

| Result | Validated workload |
|---|---|
| **16.7× lower P50 latency** | GPU exact Top-10 vs the project's scalar CPU reference on `500K × 128` vectors |
| **50% lower vector-index storage** | FP16 storage vs FP32 storage |
| **99.69% Recall@10** | FP16 storage + FP32 accumulation on `500K × 128` |
| **48/48 benchmark cases passed** | Full reproducibility matrix |
| **99.609375% minimum observed Recall@K** | Across the M15 validation matrix |

> The CPU comparison is against this project's scalar reference implementation, not an optimized production vector database. FP16 is primarily a memory-efficiency tradeoff and is not claimed to provide a universal latency improvement.

[Benchmark methodology →](docs/BENCHMARKS.md)

## Architecture

```text
Client
  │
  │ HTTP / JSON / little-endian float32 binary
  ▼
┌──────────────────────────────────────────────┐
│              Spring Boot REST API            │
│ API key • validation • request IDs • metrics │
│ capacity limits • health • error handling    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
               VortexIndex Java API
                       │
                       │ JNI
                       ▼
                  vortex_jni.dll
                       │
                       ▼
                  GpuVectorIndex
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Persistent FP32 store     Persistent FP16 store
          │                         │
          └────────────┬────────────┘
                       ▼
       CUDA exact distance + GPU Top-K kernels
                       │
                       ▼
                    NVIDIA GPU
```

[Architecture details →](docs/ARCHITECTURE.md)

## What makes it technically interesting

- **Exact GPU Top-K search** using squared-L2 distance.
- **Persistent GPU-resident indexes** instead of re-uploading vectors for every query.
- **Hierarchical and fused GPU Top-K** to avoid materializing a full production-path distance array.
- **Warp-shuffle reductions**, adaptive block sizing, and query-tile autotuning.
- **Query-tiled batched kernels** for improved amortized throughput.
- **`float4` vectorized memory access** and an explicit FP32 FMA path.
- **FP16 vector storage with FP32 accumulation** for a validated 50% storage reduction.
- **Pinned host memory + dual non-blocking CUDA streams** for asynchronous transfer/compute plumbing.
- **Nsight Compute-guided tuning** based on measured kernel bottlenecks rather than guesswork.
- **Reproducible benchmark harness** with min/mean/stddev/P50/P95/P99, QPS, Recall@1, Recall@K, CSV, and JSON output.
- **Java JNI integration** with opaque native handles, `shared_ptr` lifetime protection, and lifecycle-safe `AutoCloseable` indexes.
- **Spring Boot REST API** supporting single-query, batch-query, JSON ingestion, and binary float32 ingestion.
- **Production hardening** with API-key protection, request correlation IDs, finite-value validation, capacity guards, health checks, metrics, graceful cleanup, and exception-based CUDA failure propagation.

## Request path

```text
POST /api/v1/indexes
        │
        ▼
Create persistent GPU index
        │
        ├──────────────► FP32 storage
        └──────────────► FP16 storage

POST /api/v1/indexes/{id}/search
        │
        ▼
Spring Boot
        ▼
Java VortexIndex
        ▼
JNI
        ▼
CUDA Top-K
        ▼
JSON neighbors
```

## Precision modes

```text
FP32_EXACT
FP16_STORAGE_FP32_ACCUMULATION
```

`FP32_EXACT` is the default API mode.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/indexes` | Create a GPU index from JSON or binary float32 data |
| `GET` | `/api/v1/indexes` | List active indexes |
| `GET` | `/api/v1/indexes/{id}` | Read index statistics |
| `POST` | `/api/v1/indexes/{id}/search` | Exact single-query Top-K search |
| `POST` | `/api/v1/indexes/{id}/search/batch` | Batched exact Top-K search |
| `DELETE` | `/api/v1/indexes/{id}` | Release the index and GPU resources |
| `GET` | `/api/v1/system` | Native/CUDA runtime diagnostics |
| `GET` | `/actuator/health` | Service health |
| `GET` | `/actuator/metrics` | Operational metrics |

[REST API reference →](docs/API.md)

## Binary ingestion

For larger uploads, Vortex supports a compact binary path:

```http
POST /api/v1/indexes
Content-Type: application/octet-stream
X-Vortex-Vector-Count: <N>
X-Vortex-Dimension: <D>
```

The request body contains exactly `N × D` **little-endian IEEE-754 float32** values in vector-major order.

[Binary format →](docs/BINARY_FORMAT.md)

## Build

### Requirements

Validated development environment:

- Windows 11 x64
- NVIDIA CUDA Toolkit 13.3
- Visual Studio 2022 Build Tools
- CMake 3.24+
- JDK 17+
- Maven 3.6.3+
- NVIDIA GPU compatible with the configured architecture

The current CMake configuration targets **`sm_86`**.

### Native + Java build

```powershell
cmake `
    -S . `
    -B build `
    -G "Visual Studio 17 2022" `
    -A x64

cmake --build build --config Release --clean-first

mvn --no-transfer-progress clean package
```

Expected artifacts:

```text
build/Release/vortex_cuda.exe
build/Release/vortex_benchmark.exe
build/Release/vortex_jni.dll
server/target/vortex-server-1.0.0.jar
```

## End-to-end validation

```powershell
.\BUILD_AND_TEST_M18.ps1
```

The M18 test exercises the real native/JNI/REST stack, including:

```text
CUDA exception path
native Release build
JNI shared library
Maven + JUnit
Spring Boot startup
health/readiness
API-key rejection
binary ingestion
capacity rejection
FP32 exact search
FP16 search
batch search
input validation
metrics
delete / 404 lifecycle
```

Expected final result:

```text
============================================
 M18 PRODUCTION SMOKE TEST: PASS
============================================
```

## Run the service

```powershell
$env:VORTEX_API_KEY = "replace-with-a-strong-secret"

java `
    "-Dvortex.jni.path=D:\path\to\vortex_jni.dll" `
    -jar .\server\target\vortex-server-1.0.0.jar
```

When `VORTEX_API_KEY` is configured, `/api/**` requests require:

```text
X-Vortex-Api-Key: <secret>
```

Health probes remain available separately:

```text
GET /actuator/health
```

[Security and deployment notes →](docs/SECURITY.md)

## Release package

The tagged **v1.0.0** release includes a Windows x64 distribution.

[Download Vortex CUDA 1.0.0 →](../../releases/tag/v1.0.0)

To create the package locally from a clean worktree:

```powershell
.\PACKAGE_M18_RELEASE.ps1
```

Output:

```text
dist/vortex-cuda-1.0.0-windows-x64/
dist/vortex-cuda-1.0.0-windows-x64.zip
```

The package contains the JNI DLL, native executables, Spring Boot JAR, runtime configuration, documentation, build provenance, and a SHA-256 manifest.

## Project evolution

```text
M0–M5    CUDA exact search + GPU Top-K
M6–M12   fused, tiled, vectorized and FMA kernel optimization
M13      pinned host memory + CUDA streams
M14      FP16 storage + FP32 accumulation
M15      reproducible benchmark harness
M16      Java JNI integration
M17      Spring Boot REST backend
M18      production hardening + Vortex CUDA 1.0.0
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [REST API](docs/API.md)
- [Benchmark methodology](docs/BENCHMARKS.md)
- [Binary ingestion format](docs/BINARY_FORMAT.md)
- [Security and deployment](docs/SECURITY.md)
- [Changelog](CHANGELOG.md)

## Scope

Vortex CUDA is an exact-search engine and systems-engineering project, not a drop-in replacement for a distributed production vector database. Index data is process-resident; durable restart persistence, replication, multi-node sharding, and ANN graph/index structures are outside the 1.0.0 scope.
