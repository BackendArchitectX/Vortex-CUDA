param(
    [ValidateRange(1, 10000000)]
    [int]$Vectors = 500000,

    [ValidateRange(1, 4096)]
    [int]$Dimension = 128,

    [ValidateRange(1, 32)]
    [int]$TopK = 10,

    [ValidateSet("auto", "quick", "full")]
    [string]$Preset = "auto",

    [int[]]$BatchSizes = @(1, 8, 32),

    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
}

function Test-ContainsInt {
    param(
        [int[]]$Values,
        [int]$Value
    )

    return $Values -contains $Value
}

function Resolve-BenchmarkPreset {
    param(
        [int]$RequestedVectors,
        [int]$RequestedDimension,
        [int]$RequestedTopK,
        [string]$RequestedPreset
    )

    $quickVectors = @(100000, 500000)
    $quickDimensions = @(128, 129)
    $quickTopKs = @(10)

    $fullVectors = @(100000, 250000, 500000, 1000000)
    $fullDimensions = @(128, 129)
    $fullTopKs = @(10, 32)

    $quickCompatible =
        (Test-ContainsInt $quickVectors $RequestedVectors) -and
        (Test-ContainsInt $quickDimensions $RequestedDimension) -and
        (Test-ContainsInt $quickTopKs $RequestedTopK)

    $fullCompatible =
        (Test-ContainsInt $fullVectors $RequestedVectors) -and
        (Test-ContainsInt $fullDimensions $RequestedDimension) -and
        (Test-ContainsInt $fullTopKs $RequestedTopK)

    if ($RequestedPreset -eq "quick") {
        if (-not $quickCompatible) {
            throw "Requested workload is not part of the current M15 quick preset. Quick supports vectors={100000,500000}, dimensions={128,129}, topK={10}."
        }

        return "quick"
    }

    if ($RequestedPreset -eq "full") {
        if (-not $fullCompatible) {
            throw "Requested workload is not part of the current M15 full preset. Full supports vectors={100000,250000,500000,1000000}, dimensions={128,129}, topK={10,32}."
        }

        return "full"
    }

    if ($quickCompatible) {
        return "quick"
    }

    if ($fullCompatible) {
        return "full"
    }

    throw @"
The current M15 executable uses fixed benchmark presets and does not yet accept arbitrary workload dimensions.

Supported quick preset:
  vectors   = 100000, 500000
  dimension = 128, 129
  topK      = 10
  batches   = 1, 8, 32

Supported full preset:
  vectors   = 100000, 250000, 500000, 1000000
  dimension = 128, 129
  topK      = 10, 32
  batches   = 1, 8, 32
"@
}

function Find-BenchmarkExecutable {
    param([string]$Root)

    $candidates = @(
        (Join-Path $Root "build\Release\vortex_benchmark.exe"),
        (Join-Path $Root "dist\vortex-cuda-1.0.0-windows-x64\bin\vortex_benchmark.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw @"
vortex_benchmark.exe was not found.

Build it first from the repository root:

  cmake -S . -B build -G "Visual Studio 17 2022" -A x64
  cmake --build build --config Release
"@
}

function Format-Number {
    param(
        $Value,
        [int]$Digits = 3
    )

    return ([double]$Value).ToString("F$Digits")
}

$root = $PSScriptRoot
$selectedPreset = Resolve-BenchmarkPreset `
    -RequestedVectors $Vectors `
    -RequestedDimension $Dimension `
    -RequestedTopK $TopK `
    -RequestedPreset $Preset

$exe = Find-BenchmarkExecutable -Root $root

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $root "benchmark_demo_results"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $root $OutputDirectory
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

Write-Section "VORTEX CUDA LIVE BENCHMARK"
Write-Host ("Executable:  {0}" -f $exe)
Write-Host ("Preset:      {0}" -f $selectedPreset)
Write-Host ("Target:      {0:N0} vectors x {1}, Top-{2}" -f $Vectors, $Dimension, $TopK)
Write-Host ("Batches:     {0}" -f ($BatchSizes -join ", "))
Write-Host ("Output:      {0}" -f $OutputDirectory)

try {
    $health = Invoke-RestMethod `
        -Method Get `
        -Uri "http://localhost:8080/actuator/health" `
        -TimeoutSec 1 `
        -ErrorAction Stop

    if ($health.status -eq "UP") {
        Write-Host ""
        Write-Warning "The Vortex REST server is currently running. Stop it before benchmarking if you want cleaner, more reproducible GPU measurements."
    }
}
catch {
    # No local server detected. This is the preferred benchmark state.
}

Write-Section "RUN M15 HARNESS"
Write-Host "The native M15 harness uses deterministic seeds, warmups, measured runs,"
Write-Host "alternating FP32/FP16 order, P50/P95/P99, QPS and recall validation."
Write-Host ""

& $exe --benchmark-m15 $selectedPreset $OutputDirectory

if ($LASTEXITCODE -ne 0) {
    throw "vortex_benchmark.exe exited with code $LASTEXITCODE."
}

$summaryFile = Get-ChildItem `
    -Path $OutputDirectory `
    -Filter "vortex_m15_summary_*.csv" `
    -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $summaryFile) {
    throw "Benchmark completed but no M15 summary CSV was found in $OutputDirectory."
}

$rows = Import-Csv $summaryFile.FullName

$filtered = @(
    $rows | Where-Object {
        [int]$_.vector_count -eq $Vectors -and
        [int]$_.dimension -eq $Dimension -and
        [int]$_.top_k -eq $TopK -and
        $BatchSizes -contains [int]$_.batch_size
    }
)

if ($filtered.Count -eq 0) {
    throw "The benchmark completed, but the requested workload was not found in the generated summary CSV."
}

Write-Section "TARGET WORKLOAD RESULTS"

$resultTable = foreach ($row in ($filtered | Sort-Object { [int]$_.batch_size })) {
    [PSCustomObject]@{
        Batch          = [int]$row.batch_size
        "FP32 P50 ms"  = Format-Number $row.fp32_p50_ms 3
        "FP32 P95 ms"  = Format-Number $row.fp32_p95_ms 3
        "FP32 P99 ms"  = Format-Number $row.fp32_p99_ms 3
        "FP32 QPS"     = Format-Number $row.fp32_qps 1
        "FP16 P50 ms"  = Format-Number $row.fp16_p50_ms 3
        "FP16 P95 ms"  = Format-Number $row.fp16_p95_ms 3
        "FP16 P99 ms"  = Format-Number $row.fp16_p99_ms 3
        "FP16 QPS"     = Format-Number $row.fp16_qps 1
        "FP16/FP32"    = ((Format-Number $row.throughput_gain 3) + "x")
        "Recall@1"     = ((Format-Number $row.recall_at_1_pct 3) + "%")
        "Recall@K"     = ((Format-Number $row.recall_at_k_pct 3) + "%")
        Validation      = if ($row.validation_pass -eq "true") { "PASS" } else { "FAIL" }
    }
}

$resultTable | Format-Table -AutoSize

$first = $filtered | Select-Object -First 1
$fp32MiB = [double]$first.fp32_index_bytes / 1MB
$fp16MiB = [double]$first.fp16_index_bytes / 1MB
$reduction = [double]$first.memory_reduction_pct

Write-Section "INDEX STORAGE"
Write-Host ("FP32 vector store: {0:N2} MiB" -f $fp32MiB)
Write-Host ("FP16 vector store: {0:N2} MiB" -f $fp16MiB)
Write-Host ("Reduction:         {0:N2}%" -f $reduction)

$allPass = ($filtered | Where-Object { $_.validation_pass -ne "true" }).Count -eq 0
$minimumRecall = ($filtered | Measure-Object -Property recall_at_k_pct -Minimum).Minimum

Write-Section "VALIDATION"
Write-Host ("Minimum Recall@K: {0:N6}%" -f [double]$minimumRecall)
Write-Host ("Selected cases:   {0}" -f $filtered.Count)

if ($allPass) {
    Write-Host "Validation:       PASS" -ForegroundColor Green
}
else {
    Write-Host "Validation:       FAIL" -ForegroundColor Red
}

Write-Host ""
Write-Host "Important: this live M15 harness compares the optimized FP32 and FP16 GPU paths." -ForegroundColor Yellow
Write-Host "It does NOT rerun the historical scalar CPU reference used for the documented 16.7x P50 comparison." -ForegroundColor Yellow
Write-Host "Do not present FP16/FP32 throughput gain as CPU-vs-GPU speedup." -ForegroundColor Yellow

Write-Host ""
Write-Host ("Summary CSV: {0}" -f $summaryFile.FullName)

if ($allPass) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host " VORTEX CUDA BENCHMARK DEMO: PASS" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Red
Write-Host " VORTEX CUDA BENCHMARK DEMO: FAIL" -ForegroundColor Red
Write-Host "============================================" -ForegroundColor Red
exit 1
