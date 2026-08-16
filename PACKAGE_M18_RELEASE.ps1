param(
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

$project = $PSScriptRoot
$version = "1.0.0"
$releaseName = "vortex-cuda-$version-windows-x64"
$dist = "$project\dist"
$releaseRoot = "$dist\$releaseName"
$zipPath = "$dist\$releaseName.zip"

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

$gitStatus =
    git status --porcelain

if ($gitStatus -and
        -not $AllowDirty) {

    throw @"
Release packaging requires a clean Git worktree.
Commit or restore changes first, then rerun.

Use -AllowDirty only for a development package.
"@
}

$javacCommand = Get-Command javac -ErrorAction Stop
$javacExe = $javacCommand.Source
$javaBin = Split-Path $javacExe -Parent
$env:JAVA_HOME = Split-Path $javaBin -Parent

Write-Host "`n=== BUILD NATIVE RELEASE ==="

cmake `
    -S . `
    -B build `
    -G "Visual Studio 17 2022" `
    -A x64

cmake --build `
    build `
    --config Release `
    --clean-first

if ($LASTEXITCODE -ne 0) {
    throw "Native release build failed."
}

Write-Host "`n=== BUILD JAVA RELEASE ==="

$maven = Resolve-Maven

& $maven `
    --no-transfer-progress `
    clean `
    package

if ($LASTEXITCODE -ne 0) {
    throw "Maven release build failed."
}

$artifacts = @{
    "vortex_jni.dll" =
        "$project\build\Release\vortex_jni.dll"

    "vortex_cuda.exe" =
        "$project\build\Release\vortex_cuda.exe"

    "vortex_benchmark.exe" =
        "$project\build\Release\vortex_benchmark.exe"

    "vortex-server-$version.jar" =
        "$project\server\target\vortex-server-$version.jar"
}

foreach ($entry in $artifacts.GetEnumerator()) {
    if (-not (Test-Path $entry.Value)) {
        throw "Missing release artifact: $($entry.Value)"
    }
}

Write-Host "`n=== ASSEMBLE DISTRIBUTION ==="

if (Test-Path $releaseRoot) {
    Remove-Item `
        $releaseRoot `
        -Recurse `
        -Force
}

if (Test-Path $zipPath) {
    Remove-Item `
        $zipPath `
        -Force
}

New-Item `
    -ItemType Directory `
    -Force `
    -Path "$releaseRoot\bin" |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path "$releaseRoot\config" |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path "$releaseRoot\docs" |
    Out-Null

foreach ($entry in $artifacts.GetEnumerator()) {
    Copy-Item `
        $entry.Value `
        "$releaseRoot\bin\$($entry.Key)" `
        -Force
}

Copy-Item `
    "$project\release\application-production.yml" `
    "$releaseRoot\config\application-production.yml" `
    -Force

Copy-Item `
    "$project\release\RUN_SERVER.ps1" `
    "$releaseRoot\RUN_SERVER.ps1" `
    -Force

Copy-Item `
    "$project\README.md" `
    "$releaseRoot\README.md" `
    -Force

Copy-Item `
    "$project\CHANGELOG.md" `
    "$releaseRoot\CHANGELOG.md" `
    -Force

Copy-Item `
    "$project\docs\*.md" `
    "$releaseRoot\docs" `
    -Force

$commit =
    git rev-parse --short HEAD

@"
Vortex CUDA $version
Git commit: $commit
Build date: $(Get-Date -Format o)
Platform: Windows x64
CUDA architecture configured by CMake: sm_86
"@ |
    Set-Content `
        "$releaseRoot\BUILD_INFO.txt" `
        -Encoding UTF8

$hashLines =
    Get-ChildItem `
        $releaseRoot `
        -Recurse `
        -File |
    Where-Object {
        $_.Name -ne "SHA256SUMS.txt"
    } |
    Sort-Object FullName |
    ForEach-Object {
        $hash =
            Get-FileHash `
                $_.FullName `
                -Algorithm SHA256

        $relative =
            $_.FullName.Substring(
                $releaseRoot.Length + 1
            )

        "$($hash.Hash.ToLowerInvariant())  $relative"
    }

$hashLines |
    Set-Content `
        "$releaseRoot\SHA256SUMS.txt" `
        -Encoding ASCII

Compress-Archive `
    -Path "$releaseRoot\*" `
    -DestinationPath $zipPath `
    -CompressionLevel Optimal

Write-Host "`n=== RELEASE PACKAGE COMPLETE ==="
Write-Host "Directory: $releaseRoot"
Write-Host "ZIP:       $zipPath"
Write-Host "SHA256:    $releaseRoot\SHA256SUMS.txt"
