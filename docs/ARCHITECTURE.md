# Architecture

## Request path

```text
HTTP client
   |
   v
RequestGuardFilter
   |-- X-Request-ID
   |-- Content-Length guard
   |-- optional X-Vortex-Api-Key
   v
VortexIndexController
   v
VortexIndexService
   |-- finite-value validation
   |-- global GPU search semaphore
   |-- Micrometer metrics
   v
VortexIndexRegistry
   |-- UUID lifecycle
   |-- active-index limit
   |-- resident vector-memory budget
   v
VortexIndex (Java)
   |-- AutoCloseable
   |-- lifecycle read/write lock
   v
JNI registry
   |-- opaque long handle
   |-- shared_ptr lifetime
   |-- per-index search mutex
   v
GpuVectorIndex
   |-- persistent FP32 store
   |-- persistent FP16 store
   |-- reusable device buffers
   v
CUDA kernels
```

## Native lifetime

Java never receives a raw C++ pointer. JNI assigns an opaque handle and maps it
to a `shared_ptr<NativeIndex>`. Destroy removes the handle from the registry.
An in-flight call holding a shared pointer keeps the native index alive until
that call completes.

## Concurrency

Three layers are intentional:

1. `VortexIndex` prevents `close()` racing active Java calls.
2. JNI serializes calls that reuse one index's persistent native buffers.
3. Spring bounds total concurrent GPU search work across indexes.

## GPU memory accounting

The current engine retains both persistent vector representations:

```text
FP32 store: 4 bytes / element
FP16 store: 2 bytes / element
--------------------------------
logical persistent vector stores: 6 bytes / element
```

M18 capacity accounting covers these vector stores. Temporary buffers, CUDA
context memory and runtime overhead are not included, so production limits
should remain below physical VRAM.

## Error boundary

M18 changes `CUDA_CHECK` from `std::exit()` to `CudaException`.

```text
CUDA runtime error
    -> CudaException
    -> JNI catches std::exception
    -> Java IllegalStateException
    -> Spring HTTP 503
```

This avoids intentionally terminating the JVM for an ordinary CUDA API error.
A severe driver/GPU failure can still require process restart.

## Persistence

Indexes are in-process resources. Delete releases GPU/native resources.
Shutdown closes all remaining indexes.

Durable persistence and automatic reload are outside the 1.0.0 scope.
