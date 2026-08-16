$ErrorActionPreference = "Stop"

$project = $PSScriptRoot
$port = 18081
$baseUrl = "http://127.0.0.1:$port"
$apiKey = "m18-local-smoke-key"

Set-Location $project

function Resolve-Maven {
    $command = Get-Command mvn.cmd -ErrorAction SilentlyContinue

    if (-not $command) {
        $command = Get-Command mvn -ErrorAction SilentlyContinue
    }

    if ($command) {
        return $command.Source
    }

    throw "Maven was not found on PATH."
}

function Get-HttpStatusCodeFromError {
    param(
        [Parameter(Mandatory = $true)]
        $ErrorRecord
    )

    $response = $ErrorRecord.Exception.Response

    if ($null -eq $response) {
        return 0
    }

    try {
        return [int]$response.StatusCode
    }
    catch {
        try {
            return [int]$response.StatusCode.value__
        }
        catch {
            return 0
        }
    }
}

function Invoke-ExpectStatus {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Operation,

        [Parameter(Mandatory = $true)]
        [int]$ExpectedStatus
    )

    try {
        & $Operation | Out-Null
        throw "Expected HTTP $ExpectedStatus but request succeeded."
    }
    catch {
        $actual = Get-HttpStatusCodeFromError $_

        if ($actual -ne $ExpectedStatus) {
            throw "Expected HTTP $ExpectedStatus but received HTTP $actual. $($_.Exception.Message)"
        }
    }
}

Write-Host "`n=== M18 PRE-FLIGHT ==="

if (Select-String `
        -Path "$project\include\cuda_check.cuh" `
        -Pattern "std::exit|EXIT_FAILURE" `
        -Quiet) {

    throw "cuda_check.cuh still contains process-fatal exit behavior."
}

Write-Host "CUDA error path: exception-based"

$javacCommand = Get-Command javac -ErrorAction Stop
$javacExe = $javacCommand.Source
$javaBin = Split-Path $javacExe -Parent
$env:JAVA_HOME = Split-Path $javaBin -Parent

java -version
javac -version

if (-not (Test-Path "$env:JAVA_HOME\include\jni.h")) {
    throw "jni.h not found under JAVA_HOME=$env:JAVA_HOME"
}

Write-Host "`n=== NATIVE RELEASE BUILD ==="

cmake `
    -S . `
    -B build `
    -G "Visual Studio 17 2022" `
    -A x64

if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed."
}

cmake --build build --config Release --clean-first

if ($LASTEXITCODE -ne 0) {
    throw "Native build failed."
}

$jniDll = "$project\build\Release\vortex_jni.dll"
$benchmarkExe = "$project\build\Release\vortex_benchmark.exe"
$nativeExe = "$project\build\Release\vortex_cuda.exe"

foreach ($artifact in @(
    $jniDll,
    $benchmarkExe,
    $nativeExe
)) {
    if (-not (Test-Path $artifact)) {
        throw "Missing native artifact: $artifact"
    }
}

Write-Host "`n=== MAVEN TEST + PACKAGE ==="

$maven = Resolve-Maven

& $maven `
    --no-transfer-progress `
    clean `
    package

if ($LASTEXITCODE -ne 0) {
    throw "Maven build/test failed."
}

$jar = "$project\server\target\vortex-server-1.0.0.jar"

if (-not (Test-Path $jar)) {
    throw "Spring Boot JAR missing: $jar"
}

Write-Host "`n=== START M18 SERVER ==="

$stdout = "$project\server\target\m18-server.stdout.log"
$stderr = "$project\server\target\m18-server.stderr.log"

Remove-Item `
    $stdout, $stderr `
    -Force `
    -ErrorAction SilentlyContinue

$javaExe = (Get-Command java -ErrorAction Stop).Source

$relativeJniDll = "build\Release\vortex_jni.dll"
$relativeJar = "server\target\vortex-server-1.0.0.jar"

$serverProcess = Start-Process `
    -FilePath $javaExe `
    -WorkingDirectory $project `
    -ArgumentList @(
        "-Dvortex.jni.path=$relativeJniDll",
        "-jar",
        $relativeJar,
        "--server.port=$port",
        "--vortex.security.api-key=$apiKey",
        "--vortex.limits.max-active-indexes=1",
        "--vortex.limits.max-resident-vector-bytes=268435456",
        "--vortex.limits.max-concurrent-searches=2"
    ) `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

try {
    Write-Host "PID: $($serverProcess.Id)"

    $health = $null

    for ($attempt = 1;
         $attempt -le 60;
         $attempt++) {

        Start-Sleep -Milliseconds 500

        if ($serverProcess.HasExited) {
            Write-Host "`n=== SERVER STDOUT ==="
            Get-Content $stdout -ErrorAction SilentlyContinue

            Write-Host "`n=== SERVER STDERR ==="
            Get-Content $stderr -ErrorAction SilentlyContinue

            throw "M18 server exited during startup."
        }

        try {
            $health =
                Invoke-RestMethod `
                    -Method Get `
                    -Uri "$baseUrl/actuator/health"

            if ($health.status -eq "UP") {
                break
            }
        }
        catch {
        }
    }

    if (-not $health -or
            $health.status -ne "UP") {

        throw "Actuator health did not become UP."
    }

    Write-Host "`n=== HEALTH ==="
    $health | ConvertTo-Json -Depth 8

    Write-Host "`n=== SECURITY GUARD ==="

    Invoke-ExpectStatus `
        -ExpectedStatus 401 `
        -Operation {
            Invoke-RestMethod `
                -Method Get `
                -Uri "$baseUrl/api/v1/system"
        }

    Write-Host "Unauthenticated /api request -> 401: PASS"

    $headers = @{
        "X-Vortex-Api-Key" = $apiKey
        "X-Request-ID" = "m18-smoke-request"
    }

    $system =
        Invoke-RestMethod `
            -Method Get `
            -Uri "$baseUrl/api/v1/system" `
            -Headers $headers

    if ($system.stage -ne "M18") {
        throw "System endpoint did not report M18."
    }

    if ($system.nativeVersion -notlike "*1.0.0*") {
        throw "Native version did not report 1.0.0."
    }

    Write-Host "Native version: $($system.nativeVersion)"
    Write-Host "CUDA device:    $($system.cudaDevice.name)"
    Write-Host "Compute cap:    $($system.cudaDevice.computeCapabilityMajor).$($system.cudaDevice.computeCapabilityMinor)"

    Write-Host "`n=== BINARY INDEX INGEST ==="

    $vectorCount = 2048
    $dimension = 64
    $random = [System.Random]::new(42)

    $vectors = [float[]]::new(
        $vectorCount * $dimension
    )

    for ($index = 0;
         $index -lt $vectors.Length;
         $index++) {

        $vectors[$index] =
            [float](
                ($random.NextDouble() * 2.0) -
                1.0
            )
    }

    if (-not [BitConverter]::IsLittleEndian) {
        throw "M18 smoke script expects a little-endian host."
    }

    $binary = [byte[]]::new(
        $vectors.Length * 4
    )

    [Buffer]::BlockCopy(
        $vectors,
        0,
        $binary,
        0,
        $binary.Length
    )

    $binaryHeaders = @{
        "X-Vortex-Api-Key" = $apiKey
        "X-Request-ID" = "m18-binary-create"
        "X-Vortex-Vector-Count" = "$vectorCount"
        "X-Vortex-Dimension" = "$dimension"
    }

    $createResponse =
        Invoke-WebRequest `
            -UseBasicParsing `
            -Method Post `
            -Uri "$baseUrl/api/v1/indexes" `
            -Headers $binaryHeaders `
            -ContentType "application/octet-stream" `
            -Body $binary

    if ($createResponse.StatusCode -ne 201) {
        throw "Binary create expected HTTP 201."
    }

    if ($createResponse.Headers["X-Request-ID"] -ne
            "m18-binary-create") {

        throw "Request correlation ID was not propagated."
    }

    $created =
        $createResponse.Content |
            ConvertFrom-Json

    $indexId =
        $created.index.id

    if (-not $indexId) {
        throw "Binary create response missing index ID."
    }

    if ([math]::Abs(
            [double]$created.index.logicalMemoryReductionPercent -
            50.0
        ) -gt 0.001) {

        throw "Expected 50% logical FP16 vector-storage reduction."
    }

    Write-Host "Index ID:       $indexId"
    Write-Host "Request ID:     $($createResponse.Headers["X-Request-ID"])"
    Write-Host "FP32 bytes:     $($created.index.fp32IndexBytes)"
    Write-Host "FP16 bytes:     $($created.index.fp16IndexBytes)"

    Write-Host "`n=== CAPACITY GUARD ==="

    Invoke-ExpectStatus `
        -ExpectedStatus 507 `
        -Operation {
            Invoke-WebRequest `
                -UseBasicParsing `
                -Method Post `
                -Uri "$baseUrl/api/v1/indexes" `
                -Headers $binaryHeaders `
                -ContentType "application/octet-stream" `
                -Body $binary
        }

    Write-Host "Second index with max-active-indexes=1 -> 507: PASS"

    Write-Host "`n=== FP32 EXACT SEARCH ==="

    $query0 =
        [float[]]::new(
            $dimension
        )

    [Array]::Copy(
        $vectors,
        0,
        $query0,
        0,
        $dimension
    )

    $searchBody = @{
        query = $query0
        topK = 10
        precision = "FP32_EXACT"
    } | ConvertTo-Json -Depth 6 -Compress

    $exact =
        Invoke-RestMethod `
            -Method Post `
            -Uri "$baseUrl/api/v1/indexes/$indexId/search" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body $searchBody

    if ([int]$exact.neighbors[0].index -ne 0) {
        throw "FP32 self-query expected Top-1 index 0."
    }

    if ([math]::Abs(
            [double]$exact.neighbors[0].squaredL2Distance
        ) -gt 0.000001) {

        throw "FP32 self-query expected distance approximately zero."
    }

    Write-Host "FP32 Top-1 index 0 @ distance 0: PASS"

    Write-Host "`n=== FP16 SEARCH ==="

    $fp16Body = @{
        query = $query0
        topK = 10
        precision = "FP16_STORAGE_FP32_ACCUMULATION"
    } | ConvertTo-Json -Depth 6 -Compress

    $fp16 =
        Invoke-RestMethod `
            -Method Post `
            -Uri "$baseUrl/api/v1/indexes/$indexId/search" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body $fp16Body

    if ([int]$fp16.neighbors[0].index -ne 0) {
        throw "FP16 self-query expected Top-1 index 0."
    }

    Write-Host "FP16 Top-1 index 0: PASS"

    Write-Host "`n=== BATCH SEARCH ==="

    $query1 =
        [float[]]::new(
            $dimension
        )

    [Array]::Copy(
        $vectors,
        $dimension,
        $query1,
        0,
        $dimension
    )

    $batchQueries =
        [float[]]::new(
            2 * $dimension
        )

    [Array]::Copy(
        $query0,
        0,
        $batchQueries,
        0,
        $dimension
    )

    [Array]::Copy(
        $query1,
        0,
        $batchQueries,
        $dimension,
        $dimension
    )

    $batchBody = @{
        queries = $batchQueries
        queryCount = 2
        topK = 10
        precision = "FP32_EXACT"
    } | ConvertTo-Json -Depth 8 -Compress

    $batch =
        Invoke-RestMethod `
            -Method Post `
            -Uri "$baseUrl/api/v1/indexes/$indexId/search/batch" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body $batchBody

    if ([int]$batch.results[0][0].index -ne 0 -or
            [int]$batch.results[1][0].index -ne 1) {

        throw "Batch deterministic Top-1 validation failed."
    }

    Write-Host "Batch query 0 -> 0, query 1 -> 1: PASS"

    Write-Host "`n=== INPUT VALIDATION ==="

    $invalidBody = @{
        query = $query0
        topK = 33
        precision = "FP32_EXACT"
    } | ConvertTo-Json -Compress

    Invoke-ExpectStatus `
        -ExpectedStatus 400 `
        -Operation {
            Invoke-RestMethod `
                -Method Post `
                -Uri "$baseUrl/api/v1/indexes/$indexId/search" `
                -Headers $headers `
                -ContentType "application/json" `
                -Body $invalidBody
        }

    Write-Host "topK=33 -> 400: PASS"

    Write-Host "`n=== METRICS / SYSTEM ==="

    $systemAfterCreate =
        Invoke-RestMethod `
            -Method Get `
            -Uri "$baseUrl/api/v1/system" `
            -Headers $headers

    if ($systemAfterCreate.activeIndexes -ne 1) {
        throw "System endpoint expected one active index."
    }

    if ($systemAfterCreate.residentVectorBytes -le 0) {
        throw "System endpoint expected positive resident vector bytes."
    }

    $metrics =
        Invoke-RestMethod `
            -Method Get `
            -Uri "$baseUrl/actuator/metrics/vortex.search.requests"

    $metrics |
        ConvertTo-Json -Depth 8

    Write-Host "`n=== DELETE / 404 ==="

    Invoke-RestMethod `
        -Method Delete `
        -Uri "$baseUrl/api/v1/indexes/$indexId" `
        -Headers $headers

    Invoke-ExpectStatus `
        -ExpectedStatus 404 `
        -Operation {
            Invoke-RestMethod `
                -Method Get `
                -Uri "$baseUrl/api/v1/indexes/$indexId" `
                -Headers $headers
        }

    Write-Host "Deleted index -> 404: PASS"

    Write-Host "`n============================================"
    Write-Host " M18 PRODUCTION SMOKE TEST: PASS"
    Write-Host "============================================"
}
finally {
    if ($serverProcess -and
            -not $serverProcess.HasExited) {

        Stop-Process `
            -Id $serverProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Start-Sleep -Milliseconds 300
}

Write-Host "`n=== GIT STATUS ==="
git status
