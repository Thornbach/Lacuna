#requires -Version 5.1
<#
  Package Lacuna into a self-contained, hassle-free distribution.

    dist/Lacuna-v<ver>/        lacuna.exe + models\ (weights) + licenses
    dist/Lacuna-v<ver>.zip     the zip users download

  Users just extract and double-click lacuna.exe -- no Python, no Hugging Face,
  no tokens, no .onnx. The BURN backends load the bundled .safetensors directly.

  Variants:
    gpu  (default)  cuda / NVIDIA GPU     -> Lacuna-v<ver>-gpu.zip   (fast)
    wgpu            Vulkan/DX12/Metal GPU -> Lacuna-v<ver>-wgpu.zip  (cross-platform, no CUDA)
    cpu             ndarray, portable     -> Lacuna-v<ver>-cpu.zip   (no GPU needed, slow)

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\package.ps1 [-Variant gpu|wgpu|cpu] [-Version 0.1.0] [-SkipBuild]
#>
param(
    [string]$Version,
    [ValidateSet("gpu", "wgpu", "cpu")][string]$Variant = "gpu",
    [switch]$SkipBuild,
    # coreset_bank.bin is 879 MB -- 72% of the zip -- and only the PatchCore
    # detector uses it. A package built around the few-shot head does not need
    # it (see all_paths_ok: EITHER the head OR bank+meta is sufficient to run),
    # and dropping it takes the download from ~1.2 GB to ~340 MB.
    [switch]$NoBank,
    # Folder holding sam_encoder.onnx + sam_decoder.onnx. The Field Review SAM
    # click tool always runs on ort and has no BURN port, so unlike every other
    # model these ship as .onnx or not at all. Omit to leave SAM out.
    [string]$SamDir
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not $Version) {
    $m = Select-String -Path (Join-Path $root "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    $Version = if ($m) { $m.Matches[0].Groups[1].Value } else { "0.0.0" }
}
Write-Host "Packaging Lacuna v$Version ($Variant)" -ForegroundColor Cyan

# 1. Build. gpu = default features (cuda, self-contained); wgpu = cross-platform GPU;
#    cpu = ndarray (portable, no GPU).
if (-not $SkipBuild) {
    Write-Host "Building release ($Variant)..." -ForegroundColor Cyan
    $manifest = Join-Path $root "Cargo.toml"
    if ($Variant -eq "cpu") {
        & cargo build --release --no-default-features --manifest-path $manifest
    }
    elseif ($Variant -eq "wgpu") {
        & cargo build --release --no-default-features --features wgpu-gpu --manifest-path $manifest
    }
    else {
        & cargo build --release --manifest-path $manifest
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}
$exe = Join-Path $root "target\release\lacuna.exe"
if (-not (Test-Path $exe)) { throw "exe not found: $exe (build first, or drop -SkipBuild)" }

# 2. Staging tree.
$stage = Join-Path $root "dist\Lacuna-v$Version-$Variant"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path `
    $stage, (Join-Path $stage "models\recon"), (Join-Path $stage "LICENSES") | Out-Null

# 3. Executable.
Copy-Item $exe $stage

# 4. Runtime weights (BURN needs only these -- the big .onnx are NOT included).
$models = Join-Path $root "models"
$need = @(
    @{ src = "dino_weights.safetensors"; dst = "models\dino_weights.safetensors" },
    @{ src = "yolo_weights.safetensors"; dst = "models\yolo_weights.safetensors" },
    @{ src = "fewshot_head.json";        dst = "models\fewshot_head.json" },
    @{ src = "recon\gen.mpk";            dst = "models\recon\gen.mpk" },
    @{ src = "detector_meta.json";       dst = "models\detector_meta.json" }
)
if (-not $NoBank) {
    $need += @{ src = "coreset_bank.bin"; dst = "models\coreset_bank.bin" }
}
foreach ($f in $need) {
    $s = Join-Path $models $f.src
    if (Test-Path $s) { Copy-Item $s (Join-Path $stage $f.dst) }
    else { Write-Warning "missing weight (skipped): $($f.src)" }
}
if ($NoBank) {
    Write-Host "Skipped coreset_bank.bin (-NoBank): few-shot head only, no PatchCore." -ForegroundColor Yellow
}

# 4b. SAM (optional). Goes in models\sam\ rather than models\ so it reads as one
#     selectable unit -- the app asks for a FOLDER containing both files, not for
#     the files themselves, and nothing auto-discovers it. Whoever runs the
#     package still has to point Field Review at this folder once.
if ($SamDir) {
    $samStage = Join-Path $stage "models\sam"
    New-Item -ItemType Directory -Force -Path $samStage | Out-Null
    $samOk = $true
    foreach ($f in "sam_encoder.onnx", "sam_decoder.onnx") {
        $s = Join-Path $SamDir $f
        if (Test-Path $s) { Copy-Item $s (Join-Path $samStage $f) }
        else { Write-Warning "missing SAM model: $s"; $samOk = $false }
    }
    # A half-copied SAM folder fails at load time with a confusing error, so
    # leave nothing rather than something broken.
    if (-not $samOk) {
        Remove-Item $samStage -Recurse -Force
        Write-Warning "SAM incomplete -- models\sam\ removed from the package."
    }
    else {
        Write-Host "Bundled SAM -> models\sam\ (point Field Review's 'SAM model folder' at it)." -ForegroundColor Green
    }
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
    Write-Warning "Could not fetch DINOv3 license -- add LICENSES\DINOv3-LICENSE.md by hand before distributing."
}

# 6. Zip.
$zip = Join-Path $root "dist\Lacuna-v$Version-$Variant.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Done -> $zip ($mb MB)" -ForegroundColor Green
Write-Host "  Contents: lacuna.exe + models\ + licenses. Users extract & run lacuna.exe." -ForegroundColor Green
