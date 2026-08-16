# REST API

Base path:

```text
/api/v1
```

If `VORTEX_API_KEY` is configured, all `/api/**` requests require:

```text
X-Vortex-Api-Key: <secret>
```

Clients may send:

```text
X-Request-ID: client-request-id
```

A safe request ID is echoed in the response.

## Create index — JSON

```http
POST /api/v1/indexes
Content-Type: application/json
```

```json
{
  "vectorCount": 2,
  "dimension": 3,
  "vectors": [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0
  ]
}
```

## Create index — binary

```http
POST /api/v1/indexes
Content-Type: application/octet-stream
X-Vortex-Vector-Count: 2
X-Vortex-Dimension: 3
```

Body: six little-endian float32 values.

## Single search

```http
POST /api/v1/indexes/{id}/search
```

```json
{
  "query": [1.0, 0.0, 0.0],
  "topK": 2,
  "precision": "FP32_EXACT"
}
```

Precision values:

```text
FP32_EXACT
FP16_STORAGE_FP32_ACCUMULATION
```

## Batch search

```http
POST /api/v1/indexes/{id}/search/batch
```

```json
{
  "queries": [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0
  ],
  "queryCount": 2,
  "topK": 2,
  "precision": "FP32_EXACT"
}
```

## Other endpoints

```text
GET    /api/v1/indexes
GET    /api/v1/indexes/{id}
DELETE /api/v1/indexes/{id}
GET    /api/v1/system
```

## Error statuses

```text
400  invalid shape/value/topK
401  API key missing or invalid
404  index not found
413  request Content-Length exceeds configured limit
429  GPU search concurrency unavailable
507  active-index or vector-memory capacity exceeded
503  native CUDA/JNI engine failure
500  unexpected server failure
```

Errors use `application/problem+json` and include a request ID when available.
