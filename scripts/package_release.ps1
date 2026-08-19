#requires -Version 5.1
<#
  Assemble a release from CI artifacts.

    dist/release/Lacuna-v<ver>-models.zip              weights, ONE download for everyone
    dist/release/Lacuna-v<ver>-windows-x64-wgpu.zip    binary + licences
    dist/release/Lacuna-v<ver>-linux-x64-cpu.tar.gz    ...and so on, 8 of them

  CI already builds each per-platform archive ON the platform it targets, so
  this script does not repack them — it renames them into release form. That is
  deliberate: a .tar.gz built on Windows cannot carry a Unix executable bit
  (bsdtar has no --mode, and a Windows file has no mode to copy), so macOS and
  Linux users would hit "permission denied" on a binary that looks fine. Building
  the archive on the runner preserves the +x cargo already set.

  Models are separate because they are ~760 MB and IDENTICAL across every
  platform and both variants. Bundling them into each package meant shipping the
  same weights eight times, ~6 GB of duplicate upload for no benefit.

  The models bundle carries BOTH DINO formats on purpose:
    dino.onnx + .onnx.data      the cpu variant loads this through ONNX Runtime
    dino_weights.safetensors    the wgpu variant loads this through BURN
  One bundle therefore serves either download. Dropping the "wrong" one would
  save ~327 MB and silently break half the packages.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\package_release.ps1 `
      -RunId <ci-run-id> -Version 0.5.0 [-SamDir <path>] [-SkipDownload]
#>
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$Version = "0.5.0",
    [string]$Repo    = "Thornbach/Lacuna",
    # Folder holding sam_encoder.onnx + sam_decoder.onnx. Omit to leave SAM out
    # and save ~358 MB; Field Review's click tool is then unavailable.
    [string]$SamDir,
    [switch]$SkipDownload
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$gh   = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { $gh = "gh" }

$out = Join-Path $root "dist\release"
$dl  = Join-Path $root "dist\ci-artifacts"
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path $out | Out-Null

# CI job name -> release name. CI's default feature set IS the wgpu variant
# (wgpu-gpu + ort-backend), so unsuffixed job names map to -wgpu here. The
# *-burn rows are a build-health check for the pure-Rust fallback, not something
# anyone downloads, so they are not shipped.
$map = @{
    "windows-x64"     = "windows-x64-wgpu"
    "windows-x64-cpu" = "windows-x64-cpu"
    "linux-x64"       = "linux-x64-wgpu"
    "linux-x64-cpu"   = "linux-x64-cpu"
    "macos-arm64"     = "macos-arm64-wgpu"
    "macos-arm64-cpu" = "macos-arm64-cpu"
    "macos-x64"       = "macos-x64-wgpu"
    "macos-x64-cpu"   = "macos-x64-cpu"
}

if (-not $SkipDownload) {
    Write-Host "Downloading artifacts from run $RunId..." -ForegroundColor Cyan
    if (Test-Path $dl) { Remove-Item $dl -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dl | Out-Null
    & $gh run download $RunId --repo $Repo --dir $dl
    if ($LASTEXITCODE -ne 0) { throw "gh run download failed" }
}

foreach ($job in $map.Keys | Sort-Object) {
    $src = Join-Path $dl "lacuna-$job"
    if (-not (Test-Path $src)) { Write-Warning "artifact missing, skipped: lacuna-$job"; continue }
    $archive = Get-ChildItem $src -File | Where-Object { $_.Extension -in ".zip", ".gz" } | Select-Object -First 1
    if (-not $archive) { Write-Warning "no archive inside lacuna-$job, skipped"; continue }
    $ext = if ($archive.Name -like "*.tar.gz") { "tar.gz" } else { "zip" }
    $dst = Join-Path $out "Lacuna-v$Version-$($map[$job]).$ext"
    Copy-Item $archive.FullName $dst -Force
    Write-Host ("  {0,-38} {1,7:N1} MB" -f (Split-Path $dst -Leaf), ((Get-Item $dst).Length / 1MB)) -ForegroundColor Green
}

# ── the shared models bundle ────────────────────────────────────────────────
Write-Host "Building the models bundle..." -ForegroundColor Cyan
$mstage = Join-Path $out "stage-models"
New-Item -ItemType Directory -Force -Path (Join-Path $mstage "models\recon") | Out-Null

$models = Join-Path $root "models"
$need = @(
    "dino.onnx", "dino.onnx.data",      # cpu variant, via ONNX Runtime
    "dino_weights.safetensors",          # wgpu variant, via BURN
    "yolo_weights.safetensors",
    "fewshot_head.json",
    "detector_meta.json"
)
foreach ($f in $need) {
    $s = Join-Path $models $f
    if (Test-Path $s) { Copy-Item $s (Join-Path $mstage "models\$f") }
    else { Write-Warning "missing model (skipped): $f" }
}
$recon = Join-Path $models "recon\gen.mpk"
if (Test-Path $recon) { Copy-Item $recon (Join-Path $mstage "models\recon\gen.mpk") }

if ($SamDir) {
    $samStage = Join-Path $mstage "models\sam"
    New-Item -ItemType Directory -Force -Path $samStage | Out-Null
    $ok = $true
    foreach ($f in "sam_encoder.onnx", "sam_decoder.onnx") {
        $s = Join-Path $SamDir $f
        if (Test-Path $s) { Copy-Item $s (Join-Path $samStage $f) } else { Write-Warning "missing SAM: $f"; $ok = $false }
    }
    # A half-copied SAM folder fails at load with a confusing error; leave
    # nothing rather than something broken.
    if (-not $ok) { Remove-Item $samStage -Recurse -Force }
}

# The DINOv3 licence must travel with the weights: its terms grant
# redistribution only on condition that a copy of the agreement goes with them.
Copy-Item (Join-Path $root "THIRD_PARTY_LICENSES.md") $mstage
New-Item -ItemType Directory -Force -Path (Join-Path $mstage "LICENSES") | Out-Null
Copy-Item (Join-Path $root "LICENSES\DINOv3-LICENSE.md") (Join-Path $mstage "LICENSES")

$mzip = Join-Path $out "Lacuna-v$Version-models.zip"
Compress-Archive -Path (Join-Path $mstage "*") -DestinationPath $mzip
Remove-Item $mstage -Recurse -Force
Write-Host ("  {0,-38} {1,7:N1} MB" -f (Split-Path $mzip -Leaf), ((Get-Item $mzip).Length / 1MB)) -ForegroundColor Green

Write-Host "`nDone -> $out" -ForegroundColor Green
Get-ChildItem $out -File | Sort-Object Name |
    Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
