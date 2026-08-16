# Benchmark methodology and validated results

The M15 harness writes:

```text
vortex_m15_summary_<run-id>.csv
vortex_m15_raw_<run-id>.csv
vortex_m15_report_<run-id>.json
```

Generated benchmark artifacts are ignored by Git.

## Validated development hardware

```text
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Compute capability: 8.6
CMake target: sm_86
```

## Methodology

The harness uses deterministic seeds, warmups, measured iterations,
alternating FP32/FP16 execution order, P50/P95/P99, mean/stddev, P50-derived
QPS, Recall@1/Recall@K validation and Git/CUDA metadata.

## Fused exact Top-K

For `500K x 128`, Top-10, the fused GPU path achieved about **16.7x lower P50
latency** than the project's scalar CPU exact reference.

This is not a comparison against an optimized commercial/production vector
database.

## Query-tiled batching

For `500K x 128`, Top-10, batch 16, M8 measured approximately:

```text
P50 batch: 10.522 ms
amortized: 0.658 ms/query
P50-derived QPS: ~1521
gain vs previous batched path: ~2.42x
```

## FP16 storage

For `500K x 128`:

```text
FP32 vector store: ~244.14 MiB
FP16 vector store: ~122.07 MiB
logical reduction: 50%
Recall@1: 100%
Recall@10: 99.6875%
```

FP16 is not claimed to be universally faster. Some large-batch workloads were
near-neutral or slightly slower.

## M15 full matrix

The full matrix completed **48/48 validation cases**. The lowest observed
Recall@K was **99.609375%**.

Re-run the harness on the deployment machine before capacity planning because
thermals, power state, driver/toolkit versions and competing GPU load affect
results.
