$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$jar = "$root\bin\vortex-server-1.0.0.jar"
$jniDll = "$root\bin\vortex_jni.dll"

if (-not (Test-Path $jar)) {
    throw "Missing server JAR: $jar"
}

if (-not (Test-Path $jniDll)) {
    throw "Missing JNI DLL: $jniDll"
}

if ([string]::IsNullOrWhiteSpace(
        $env:VORTEX_API_KEY
    ) -and
        $env:VORTEX_ALLOW_INSECURE -ne
            "true") {

    throw @"
VORTEX_API_KEY is not set.

Secured run:
  `$env:VORTEX_API_KEY = "replace-with-a-strong-secret"

Local-only insecure run:
  `$env:VORTEX_ALLOW_INSECURE = "true"
"@
}

Push-Location $root

try {
    & java `
        "-Dvortex.jni.path=$jniDll" `
        -jar $jar `
        "--spring.profiles.active=production" `
        "--spring.config.additional-location=optional:file:./config/"
}
finally {
    Pop-Location
}
