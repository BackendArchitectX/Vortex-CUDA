$ErrorActionPreference = "Stop"

$project = "D:\Github OSC\Vortex CUDA\Cuda"
$port = 18080
$baseUrl = "http://127.0.0.1:$port"

Set-Location $project

function Resolve-Maven {
    $command = Get-Command mvn.cmd -ErrorAction SilentlyContinue

    if (-not $command) {
        $command = Get-Command mvn -ErrorAction SilentlyContinue
    }

    if ($command) {
        return $command.Source
    }

    $patterns = @(
        "$env:ProgramFiles\JetBrains\IntelliJ IDEA*\plugins\maven\lib\maven3\bin\mvn.cmd",
        "$env:LOCALAPPDATA\Programs\IntelliJ IDEA*\plugins\maven\lib\maven3\bin\mvn.cmd",
        "$env:USERPROFILE\AppData\Local\JetBrains\Toolbox\apps\IDEA-*\*\*\plugins\maven\lib\maven3\bin\mvn.cmd"
    )

    foreach ($pattern in $patterns) {
        $candidate = Get-ChildItem $pattern -ErrorAction SilentlyContinue |
            Select-Object -First 1

        if ($candidate) {
            return $candidate.FullName
        }
    }

    throw "Maven was not found. Configure Maven 3.6.3+ in PATH or use IntelliJ's bundled Maven."
}

Write-Host "`n=== JAVA / JNI ==="

$javacCommand = Get-Command javac -ErrorAction Stop
$javacExe = $javacCommand.Source
$javaBin = Split-Path $javacExe -Parent
$env:JAVA_HOME = Split-Path $javaBin -Parent

if (-not (Test-Path "$env:JAVA_HOME\include\jni.h")) {
    throw "jni.h not found under JAVA_HOME=$env:JAVA_HOME"
}

java -version
javac -version

Write-Host "JAVA_HOME=$env:JAVA_HOME"

Write-Host "`n=== NATIVE RELEASE BUILD ==="

cmake `
    -S . `
    -B build `
    -G "Visual Studio 17 2022" `
    -A x64

if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed."
}

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    throw "Native build failed."
}

$jniDll = "$project\build\Release\vortex_jni.dll"

if (-not (Test-Path $jniDll)) {
    throw "JNI DLL missing: $jniDll"
}

Write-Host "`n=== MAVEN BUILD ==="

$maven = Resolve-Maven
Write-Host "Maven: $maven"

& $maven -version

& $maven `
    --no-transfer-progress `
    clean `
    package

if ($LASTEXITCODE -ne 0) {
    throw "Maven build failed."
}

$jar = "$project\server\target\vortex-server-0.4.0-SNAPSHOT.jar"

if (-not (Test-Path $jar)) {
    throw "Spring Boot JAR missing: $jar"
}

Write-Host "`n=== START SPRING BOOT SERVER ==="

$stdout = "$project\server\target\m17-server.stdout.log"
$stderr = "$project\server\target\m17-server.stderr.log"

Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue

$javaExe = (Get-Command java -ErrorAction Stop).Source

$relativeJniDll = "build\Release\vortex_jni.dll"
$relativeJar = "server\target\vortex-server-0.4.0-SNAPSHOT.jar"

$serverProcess = Start-Process `
    -FilePath $javaExe `
    -WorkingDirectory $project `
    -ArgumentList @(
        "-Dvortex.jni.path=$relativeJniDll",
        "-jar",
        $relativeJar,
        "--server.port=$port"
    ) `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

try {
    Write-Host "PID: $($serverProcess.Id)"

    $health = $null

    for ($attempt = 1; $attempt -le 60; $attempt++) {
        Start-Sleep -Milliseconds 500

        if ($serverProcess.HasExited) {
            Write-Host "`n=== SERVER STDOUT ==="
            Get-Content $stdout -ErrorAction SilentlyContinue

            Write-Host "`n=== SERVER STDERR ==="
            Get-Content $stderr -ErrorAction SilentlyContinue

            throw "Spring Boot server exited during startup."
        }

        try {
            $health = Invoke-RestMethod `
                -Method Get `
                -Uri "$baseUrl/actuator/health"

            if ($health.status -eq "UP") {
                break
            }
        }
        catch {
            # Continue polling while Tomcat starts.
        }
    }

    if (-not $health -or $health.status -ne "UP") {
        Write-Host "`n=== SERVER STDOUT ==="
        Get-Content $stdout -ErrorAction SilentlyContinue

        Write-Host "`n=== SERVER STDERR ==="
        Get-Content $stderr -ErrorAction SilentlyContinue

        throw "Actuator health did not become UP."
    }

    Write-Host "`n=== HEALTH ==="
    $health | ConvertTo-Json -Depth 10

    Write-Host "`n=== CREATE DETERMINISTIC TEST INDEX ==="

    $vectorCount = 1024
    $dimension = 64
    $random = [System.Random]::new(42)

    $vectors = [float[]]::new($vectorCount * $dimension)

    for ($i = 0; $i -lt $vectors.Length; $i++) {
        $vectors[$i] = [float](($random.NextDouble() * 2.0) - 1.0)
    }

    # Query 0 exactly equals database vector 0, so exact Top-1 must be index 0.
    $query0 = [float[]]::new($dimension)
    [Array]::Copy($vectors, 0, $query0, 0, $dimension)

    $createBody = @{
        vectorCount = $vectorCount
        dimension = $dimension
        vectors = $vectors
    } | ConvertTo-Json -Depth 6 -Compress

    $created = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/api/v1/indexes" `
        -ContentType "application/json" `
        -Body $createBody

    $indexId = $created.index.id

    if (-not $indexId) {
        throw "Create index response did not contain an ID."
    }

    Write-Host "Index ID: $indexId"
    $created | ConvertTo-Json -Depth 10

    if ([math]::Abs([double]$created.index.logicalMemoryReductionPercent - 50.0) -gt 0.001) {
        throw "Expected approximately 50% logical FP16 storage reduction."
    }

    Write-Host "`n=== FP32 EXACT SEARCH ==="

    $searchBody = @{
        query = $query0
        topK = 10
        precision = "FP32_EXACT"
    } | ConvertTo-Json -Depth 6 -Compress

    $exact = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/api/v1/indexes/$indexId/search" `
        -ContentType "application/json" `
        -Body $searchBody

    $exact | ConvertTo-Json -Depth 10

    if ($exact.neighbors.Count -ne 10) {
        throw "Expected 10 exact neighbors."
    }

    if ([int]$exact.neighbors[0].index -ne 0) {
        throw "Expected exact Top-1 index 0 for self-query."
    }

    if ([math]::Abs([double]$exact.neighbors[0].squaredL2Distance) -gt 0.000001) {
        throw "Expected exact Top-1 distance approximately zero."
    }

    Write-Host "`n=== FP16 MIXED-PRECISION SEARCH ==="

    $fp16Body = @{
        query = $query0
        topK = 10
        precision = "FP16_STORAGE_FP32_ACCUMULATION"
    } | ConvertTo-Json -Depth 6 -Compress

    $fp16 = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/api/v1/indexes/$indexId/search" `
        -ContentType "application/json" `
        -Body $fp16Body

    if ($fp16.neighbors.Count -ne 10) {
        throw "Expected 10 FP16 neighbors."
    }

    Write-Host "FP16 Top-1 index: $($fp16.neighbors[0].index)"

    Write-Host "`n=== BATCH SEARCH ==="

    $query1 = [float[]]::new($dimension)
    [Array]::Copy($vectors, $dimension, $query1, 0, $dimension)

    $batchQueries = [float[]]::new($dimension * 2)
    [Array]::Copy($query0, 0, $batchQueries, 0, $dimension)
    [Array]::Copy($query1, 0, $batchQueries, $dimension, $dimension)

    $batchBody = @{
        queries = $batchQueries
        queryCount = 2
        topK = 10
        precision = "FP32_EXACT"
    } | ConvertTo-Json -Depth 8 -Compress

    $batch = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/api/v1/indexes/$indexId/search/batch" `
        -ContentType "application/json" `
        -Body $batchBody

    if ($batch.queryCount -ne 2 -or $batch.results.Count -ne 2) {
        throw "Batch search shape validation failed."
    }

    if ([int]$batch.results[0][0].index -ne 0) {
        throw "Batch query 0 expected Top-1 index 0."
    }

    if ([int]$batch.results[1][0].index -ne 1) {
        throw "Batch query 1 expected Top-1 index 1."
    }

    Write-Host "Batch query 0 Top-1: $($batch.results[0][0].index)"
    Write-Host "Batch query 1 Top-1: $($batch.results[1][0].index)"

    Write-Host "`n=== LIST / STATS ==="

    $list = Invoke-RestMethod `
        -Method Get `
        -Uri "$baseUrl/api/v1/indexes"

    $stats = Invoke-RestMethod `
        -Method Get `
        -Uri "$baseUrl/api/v1/indexes/$indexId"

    if ($list.count -lt 1) {
        throw "Index list should contain at least one index."
    }

    if ($stats.id -ne $indexId) {
        throw "Stats endpoint returned the wrong index."
    }

    Write-Host "Active indexes: $($list.count)"

    Write-Host "`n=== CUSTOM METRICS ==="

    $metrics = Invoke-RestMethod `
        -Method Get `
        -Uri "$baseUrl/actuator/metrics/vortex.search.requests"

    $metrics | ConvertTo-Json -Depth 10

    Write-Host "`n=== DELETE INDEX ==="

    Invoke-RestMethod `
        -Method Delete `
        -Uri "$baseUrl/api/v1/indexes/$indexId"

    $deletedIs404 = $false

    try {
        Invoke-RestMethod `
            -Method Get `
            -Uri "$baseUrl/api/v1/indexes/$indexId"
    }
    catch {
        if ($_.Exception.Response.StatusCode.value__ -eq 404) {
            $deletedIs404 = $true
        }
    }

    if (-not $deletedIs404) {
        throw "Deleted index should return HTTP 404."
    }

    Write-Host "Deleted index returns 404: PASS"

    Write-Host "`n============================================"
    Write-Host " M17 REST SMOKE TEST: PASS"
    Write-Host "============================================"
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process `
            -Id $serverProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Start-Sleep -Milliseconds 300
}

Write-Host "`n=== GIT STATUS ==="
git status

