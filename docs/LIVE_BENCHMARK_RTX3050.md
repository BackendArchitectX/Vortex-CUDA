# RTX 3050 live benchmark snapshot

This snapshot records repeated live-demo measurements from the Vortex CUDA
`v1.0.0` native benchmark binary on the development laptop.

## Hardware and build

```text
GPU:                NVIDIA GeForce RTX 3050 6GB Laptop GPU
Compute capability: 8.6
CUDA runtime:       13.1
CUDA driver API:    13.1
Native benchmark:   Vortex CUDA v1.0.0 build
Embedded commit:    a8571fa
Database seed:      42
Query seed:         1337
Warmups/case:       3
Measured runs/case: 15
```

## Target workload

```text
Vectors:   500,000
Dimension: 128
Metric:    exact squared L2
Top-K:     10
Batches:   1, 8, 32
```

## Repeated live-demo range

Five repeated runs produced the following target-workload ranges:

| Batch | FP32 P50 | FP16 P50 | FP32 QPS | FP16 QPS | Recall@K |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.671–1.723 ms | 0.987–1.012 ms | 580–599 | 988–1,013 | 100% |
| 8 | 4.874–4.939 ms | 4.850–4.901 ms | 1,620–1,641 | 1,632–1,649 | 100% |
| 32 | 19.256–19.503 ms | 19.553–19.804 ms | 1,641–1,662 | 1,616–1,637 | 99.6875% |

A representative run (`20260818_015337`) reported:

| Batch | FP32 P50 | FP32 P95 | FP32 P99 | FP32 QPS | FP16 P50 | FP16 P95 | FP16 P99 | FP16 QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.723 ms | 1.789 ms | 1.805 ms | 580.2 | 1.002 ms | 1.162 ms | 1.163 ms | 997.8 |
| 8 | 4.874 ms | 4.901 ms | 4.902 ms | 1,641.4 | 4.850 ms | 4.922 ms | 4.934 ms | 1,649.5 |
| 32 | 19.340 ms | 20.027 ms | 20.295 ms | 1,654.6 | 19.658 ms | 20.406 ms | 20.557 ms | 1,627.9 |

## Storage and precision

```text
FP32 vector store: 244.14 MiB
FP16 vector store: 122.07 MiB
Logical reduction: 50.00%
Recall@1:          100%
Minimum Recall@K:  99.6875%
Validation:        PASS
```

## Interpretation

The single-query FP16-storage path is materially faster on this workload,
while the higher-batch FP16 path becomes approximately throughput-neutral or
slightly slower. Vortex therefore treats FP16 primarily as a memory-efficiency
option rather than claiming a universal latency improvement.

FP32 batching scales from roughly 580–599 QPS at batch 1 to roughly
1,641–1,662 QPS at batch 32, demonstrating higher amortized throughput before
the workload approaches its observed throughput ceiling on this GPU.

## Important caveat

These runs were intentionally captured during the live demo while the Vortex
Spring Boot REST server was still listening on port 8080. They demonstrate
repeatability under the complete local demo environment, but they are **not**
the isolated benchmark to use for hardware capacity planning.

The historical **16.7x lower P50 latency** result is a different comparison:
it compares an earlier optimized GPU exact Top-10 path against this project's
scalar CPU exact reference on `500K x 128`. The FP32/FP16 numbers in this file
must not be presented as CPU-vs-GPU speedup.

Run `DEMO_BENCHMARK.ps1` with competing GPU/service workloads stopped when a
clean isolated snapshot is required.
