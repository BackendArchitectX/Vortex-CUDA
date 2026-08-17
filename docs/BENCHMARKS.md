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

## Repeated live-demo benchmark — 2026-08-18

The `DEMO_BENCHMARK.ps1` wrapper was run repeatedly against the native
`v1.0.0` benchmark executable for `500K x 128`, exact Top-10.

Across five repeated live-demo runs:

| Batch | FP32 P50 | FP16 P50 | FP32 QPS | FP16 QPS | Recall@K |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.671–1.723 ms | 0.987–1.012 ms | 580–599 | 988–1,013 | 100% |
| 8 | 4.874–4.939 ms | 4.850–4.901 ms | 1,620–1,641 | 1,632–1,649 | 100% |
| 32 | 19.256–19.503 ms | 19.553–19.804 ms | 1,641–1,662 | 1,616–1,637 | 99.6875% |

A representative run (`20260818_015337`) measured:

```text
Batch 1  FP32 P50: 1.723 ms   FP16 P50: 1.002 ms
Batch 8  FP32 P50: 4.874 ms   FP16 P50: 4.850 ms
Batch 32 FP32 P50: 19.340 ms  FP16 P50: 19.658 ms
Batch 32 FP32 QPS: 1654.6     FP16 QPS: 1627.9
```

The live runs also reproduced:

```text
FP32 vector store: 244.14 MiB
FP16 vector store: 122.07 MiB
logical reduction: 50%
Recall@1: 100%
minimum Recall@K: 99.6875%
validation: PASS
```

These live-demo runs were captured while the Spring Boot service was still
running locally. They demonstrate repeatability in the complete demo
environment, but they should not be treated as an isolated capacity-planning
benchmark. See [`LIVE_BENCHMARK_RTX3050.md`](LIVE_BENCHMARK_RTX3050.md) for the
full snapshot and caveats.

## M15 full matrix

The full matrix completed **48/48 validation cases**. The lowest observed
Recall@K was **99.609375%**.

Re-run the harness on the deployment machine before capacity planning because
thermals, power state, driver/toolkit versions and competing GPU load affect
results.
