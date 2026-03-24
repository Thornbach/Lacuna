#requires -Version 5.1
<#
  Package Lacuna into a self-contained, hassle-free distribution.

    dist/Lacuna-v<ver>/        lacuna.exe + models\ (weights) + licenses
    dist/Lacuna-v<ver>.zip     the zip users download

  Users just extract and double-click lacuna.exe — no Python, no Hugging Face,
  no tokens, no .onnx. The BURN backends load the bundled .safetensors directly.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\package.ps1 [-Version 0.1.0] [-SkipBuild]
#>
param(
    [string]$Version,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not $Version) {
    $m = Select-String -Path (Join-Path $root "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    $Version = if ($m) { $m.Matches[0].Groups[1].Value } else { "0.0.0" }
}
Write-Host "Packaging Lacuna v$Version" -ForegroundColor Cyan

# 1. Build (release, default features = cuda → self-contained, no onnxruntime).
if (-not $SkipBuild) {
    Write-Host "Building release..." -ForegroundColor Cyan
    & cargo build --release --manifest-path (Join-Path $root "Cargo.toml")
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}
$exe = Join-Path $root "target\release\lacuna.exe"
if (-not (Test-Path $exe)) { throw "exe not found: $exe (build first, or drop -SkipBuild)" }

# 2. Staging tree.
$stage = Join-Path $root "dist\Lacuna-v$Version"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path `
    $stage, (Join-Path $stage "models\recon"), (Join-Path $stage "LICENSES") | Out-Null

# 3. Executable.
Copy-Item $exe $stage

# 4. Runtime weights (BURN needs only these — the big .onnx are NOT included).
$models = Join-Path $root "models"
$need = @(
    @{ src = "dino_weights.safetensors"; dst = "models\dino_weights.safetensors" },
    @{ src = "yolo_weights.safetensors"; dst = "models\yolo_weights.safetensors" },
    @{ src = "fewshot_head.json";        dst = "models\fewshot_head.json" },
    @{ src = "recon\gen.mpk";            dst = "models\recon\gen.mpk" }
)
foreach ($f in $need) {
    $s = Join-Path $models $f.src
    if (Test-Path $s) { Copy-Item $s (Join-Path $stage $f.dst) }
    else { Write-Warning "missing weight (skipped): $($f.src)" }
}

# 5. Docs + licenses.
foreach ($doc in "README.md", "THIRD_PARTY_LICENSES.md", "LICENSE") {
    $s = Join-Path $root $doc
    if (Test-Path $s) { Copy-Item $s $stage }
}
# Fetch the authoritative DINOv3 license (its terms require bundling a copy).
try {
    Invoke-WebRequest "https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md" `
        -OutFile (Join-Path $stage "LICENSES\DINOv3-LICENSE.md") -UseBasicParsing
    Write-Host "Fetched DINOv3 license." -ForegroundColor Green
}
catch {
    Write-Warning "Could not fetch DINOv3 license — add LICENSES\DINOv3-LICENSE.md by hand before distributing."
}

# 6. Zip.
$zip = Join-Path $root "dist\Lacuna-v$Version.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Done → $zip ($mb MB)" -ForegroundColor Green
Write-Host "  Contents: lacuna.exe + models\ + licenses. Users extract & run lacuna.exe." -ForegroundColor Green
