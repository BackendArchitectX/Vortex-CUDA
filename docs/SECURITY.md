# Security and production deployment

## API key

Set:

```powershell
$env:VORTEX_API_KEY = "a-long-random-secret"
```

Then `/api/**` requires:

```text
X-Vortex-Api-Key: <secret>
```

Comparison uses `MessageDigest.isEqual`. The key is never written to request
logs.

## Request correlation

Clients may send `X-Request-ID`. Safe IDs are returned in the response and
placed in SLF4J MDC.

## Capacity protection

M18 adds:

- Content-Length request guard
- vector count/dimension/element limits
- active-index limit
- resident persistent-vector memory budget
- global concurrent search permits
- bounded search-permit wait

The application Content-Length check is only one layer. A reverse proxy/load
balancer should also enforce body limits.

## Actuator

Health/info/metrics are outside the `/api/**` API-key filter so orchestrators
can probe the service. In a real deployment, expose management endpoints only
on a trusted network.

Health details are disabled.

## CUDA failures

CUDA runtime failures become C++ exceptions rather than calling `std::exit`.
JNI translates native failures to Java exceptions and REST returns HTTP 503
without returning native file paths to clients.

A fatal driver reset may still require restarting the service process.
