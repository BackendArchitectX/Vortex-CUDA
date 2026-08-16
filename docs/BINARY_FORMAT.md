# Binary vector ingestion

M18 adds a lower-overhead index creation path.

## Contract

```text
POST /api/v1/indexes
Content-Type: application/octet-stream
X-Vortex-Vector-Count: N
X-Vortex-Dimension: D
```

Body size:

```text
N * D * 4 bytes
```

## Encoding

Each scalar is:

- IEEE-754 float32
- little-endian
- vector-major, dimension-contiguous

Two 3D vectors are encoded as:

```text
v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]
```

The service rejects size mismatches, non-finite values and configured capacity
violations.

The JNI 1.0.0 boundary still receives Java `float[]`; a direct-buffer native
ingestion API is a future optimization.
