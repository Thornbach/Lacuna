#requires -Version 5.1
<#
  Assemble the v0.5.0 release from CI artifacts.

    dist/release/Lacuna-v<ver>-models.zip            weights, ONE download for everyone
    dist/release/Lacuna-v<ver>-windows-x64-wgpu.zip  binary + licences
    dist/release/Lacuna-v<ver>-linux-x64-cpu.tar.gz  ...and so on, 8 of them

  Why models are separate: they are ~760 MB and IDENTICAL across every platform
  and both variants. Bundling them into each package meant shipping the same
  weights eight times, ~6 GB of duplicate upload for no benefit.

  The models bundle carries BOTH DINO formats on purpose:
    dino.onnx + .onnx.data      the cpu variant loads this through ONNX Runtime
    dino_weights.safetensors    the wgpu variant loads this through BURN
  One bundle therefore serves either download. Dropping the "wrong" one would
  save ~327 MB and silently break half the packages.

  Format differs by platform deliberately. Windows gets .zip; macOS and Linux get
  .tar.gz because a zip written on Windows does not preserve the Unix executable
  bit, and the first thing a user would hit is "permission denied" on a binary
  that looks fine.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\package_release.ps1 `
      -RunId 32245601387 -Version 0.5.0 [-SamDir <path>] [-SkipDownload]
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
New-Item -ItemType Directory -Force -Path $out | Out-Null

# ── CI artifact name -> what we call it in the release ──────────────────────
# CI's default feature set IS the wgpu variant (wgpu-gpu + ort-backend), so the
# unsuffixed job names map to -wgpu here.
$targets = @(
    @{ ci = "lacuna-windows-x64";       name = "windows-x64-wgpu"; archive = "zip"; bin = "lacuna.exe" },
    @{ ci = "lacuna-windows-x64-cpu";   name = "windows-x64-cpu";  archive = "zip"; bin = "lacuna.exe" },
    @{ ci = "lacuna-linux-x64";         name = "linux-x64-wgpu";   archive = "tar"; bin = "lacuna" },
    @{ ci = "lacuna-linux-x64-cpu";     name = "linux-x64-cpu";    archive = "tar"; bin = "lacuna" },
    @{ ci = "lacuna-macos-arm64";       name = "macos-arm64-wgpu"; archive = "tar"; bin = "lacuna" },
    @{ ci = "lacuna-macos-arm64-cpu";   name = "macos-arm64-cpu";  archive = "tar"; bin = "lacuna" },
    @{ ci = "lacuna-macos-x64";         name = "macos-x64-wgpu";   archive = "tar"; bin = "lacuna" },
    @{ ci = "lacuna-macos-x64-cpu";     name = "macos-x64-cpu";    archive = "tar"; bin = "lacuna" }
)

if (-not $SkipDownload) {
    Write-Host "Downloading artifacts from run $RunId..." -ForegroundColor Cyan
    if (Test-Path $dl) { Remove-Item $dl -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dl | Out-Null
    & $gh run download $RunId --repo $Repo --dir $dl
    if ($LASTEXITCODE -ne 0) { throw "gh run download failed" }
}

# ── docs that travel with every binary ──────────────────────────────────────
# LICENSE is not optional: Lacuna is AGPL-3.0 because it incorporates a model
# trained with Ultralytics YOLO, and handing a binary to someone is exactly the
# distribution that triggers the obligation. Fail loudly rather than ship without.
$docs = @("README.md", "THIRD_PARTY_LICENSES.md", "LICENSE")
foreach ($d in $docs) {
    if (-not (Test-Path (Join-Path $root $d))) { throw "missing required document: $d" }
}

foreach ($t in $targets) {
    $src = Join-Path $dl $t.ci
    if (-not (Test-Path $src)) { Write-Warning "artifact missing, skipped: $($t.ci)"; continue }
    $binSrc = Join-Path $src $t.bin
    if (-not (Test-Path $binSrc)) { Write-Warning "binary missing in $($t.ci), skipped"; continue }

    $stage = Join-Path $out "stage-$($t.name)"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Join-Path $stage "LICENSES") | Out-Null

    Copy-Item $binSrc $stage
    foreach ($d in $docs) { Copy-Item (Join-Path $root $d) $stage }
    $dino = Join-Path $root "dist\Lacuna-v$Version-cpu\LICENSES\DINOv3-LICENSE.md"
    if (Test-Path $dino) { Copy-Item $dino (Join-Path $stage "LICENSES") }

    # Windows also needs DirectML.dll beside the exe: ort-sys links it
    # dynamically and it is only an inbox component from Win10 1903 onward.
    # Without it the process dies silently (windows_subsystem = "windows").
    if ($t.archive -eq "zip") {
        $dml = Join-Path $root "target-cpu\release\DirectML.dll"
        if (Test-Path $dml) { Copy-Item $dml $stage }
        else { Write-Warning "DirectML.dll not found -- $($t.name) relies on the system copy" }
    }

    $base = "Lacuna-v$Version-$($t.name)"
    if ($t.archive -eq "zip") {
        $dst = Join-Path $out "$base.zip"
        if (Test-Path $dst) { Remove-Item $dst -Force }
        Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $dst
    }
    else {
        # tar preserves the executable bit; Compress-Archive does not, and a
        # macOS/Linux user would hit "permission denied" on an otherwise fine
        # binary. bsdtar ships with Windows 10+.
        $dst = Join-Path $out "$base.tar.gz"
        if (Test-Path $dst) { Remove-Item $dst -Force }
        Push-Location $stage
        try { & tar --create --gzip --file $dst --mode='a+x' * } finally { Pop-Location }
    }
    Remove-Item $stage -Recurse -Force
    Write-Host ("  {0,-34} {1,7:N1} MB" -f (Split-Path $dst -Leaf), ((Get-Item $dst).Length / 1MB)) -ForegroundColor Green
}

# ── the shared models bundle ────────────────────────────────────────────────
Write-Host "Building the models bundle..." -ForegroundColor Cyan
$mstage = Join-Path $out "stage-models"
if (Test-Path $mstage) { Remove-Item $mstage -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $mstage "models\recon") | Out-Null

$models = Join-Path $root "models"
$need = @(
    "dino.onnx", "dino.onnx.data",     # cpu variant, via ONNX Runtime
    "dino_weights.safetensors",         # wgpu variant, via BURN
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

Copy-Item (Join-Path $root "THIRD_PARTY_LICENSES.md") $mstage
$dino = Join-Path $root "dist\Lacuna-v$Version-cpu\LICENSES\DINOv3-LICENSE.md"
if (Test-Path $dino) {
    New-Item -ItemType Directory -Force -Path (Join-Path $mstage "LICENSES") | Out-Null
    Copy-Item $dino (Join-Path $mstage "LICENSES")
}

$mzip = Join-Path $out "Lacuna-v$Version-models.zip"
if (Test-Path $mzip) { Remove-Item $mzip -Force }
Compress-Archive -Path (Join-Path $mstage "*") -DestinationPath $mzip
Remove-Item $mstage -Recurse -Force
Write-Host ("  {0,-34} {1,7:N1} MB" -f (Split-Path $mzip -Leaf), ((Get-Item $mzip).Length / 1MB)) -ForegroundColor Green

Write-Host "`nDone -> $out" -ForegroundColor Green
Get-ChildItem $out -File | Sort-Object Name |
    Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
