#requires -Version 5.1
<#
  Package Lacuna into a self-contained, hassle-free distribution.

    dist/Lacuna-v<ver>/        lacuna.exe + models\ (weights) + licenses
    dist/Lacuna-v<ver>.zip     the zip users download

  Users just extract and double-click lacuna.exe -- no Python, no Hugging Face,
  no tokens, no separate runtime to install. The GPU variants load the bundled
  .safetensors through BURN; the cpu variant loads dino.onnx through ONNX
  Runtime, which is STATICALLY linked into lacuna.exe (there is no
  onnxruntime.dll to ship).

  Variants:
    gpu  (default)  cuda / NVIDIA GPU     -> Lacuna-v<ver>-gpu.zip   (fast)
    wgpu            Vulkan/DX12/Metal GPU -> Lacuna-v<ver>-wgpu.zip  (cross-platform, no CUDA)
    cpu             ndarray + ort CPU EP  -> Lacuna-v<ver>-cpu.zip   (no GPU needed)

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
#    cpu = ndarray + ort CPU EP (portable, no GPU).
#
# The cpu variant builds into its OWN target dir. Alternating feature sets in a
# shared target dir forces a full backend recompile each way (~10-11 min), so
# packaging cpu used to silently evict a warm wgpu/cuda tree from target\.
$targetDir = if ($Variant -eq "cpu") { Join-Path $root "target-cpu" } else { Join-Path $root "target" }
if (-not $SkipBuild) {
    Write-Host "Building release ($Variant) -> $targetDir" -ForegroundColor Cyan
    $manifest = Join-Path $root "Cargo.toml"
    if ($Variant -eq "cpu") {
        # ort-backend is REQUIRED here, not optional. use_ort() is itself
        # #[cfg(feature = "ort-backend")], so with the feature off the whole ort
        # dispatch is compiled out and DINO silently runs on burn ndarray --
        # measured 4114 ms/tile vs ort's 545 ms/tile at res 512, a 7.1x penalty
        # on the one build that can least afford it. The ort runtime is
        # statically linked (cargo:rustc-link-lib=static=onnxruntime), so this
        # adds no DLL and no external dependency.
        & cargo build --release --no-default-features --features ort-backend --target-dir $targetDir --manifest-path $manifest
    }
    elseif ($Variant -eq "wgpu") {
        & cargo build --release --no-default-features --features wgpu-gpu --target-dir $targetDir --manifest-path $manifest
    }
    else {
        & cargo build --release --target-dir $targetDir --manifest-path $manifest
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}
$exe = Join-Path $targetDir "release\lacuna.exe"
if (-not (Test-Path $exe)) { throw "exe not found: $exe (build first, or drop -SkipBuild)" }

# 2. Staging tree.
$stage = Join-Path $root "dist\Lacuna-v$Version-$Variant"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path `
    $stage, (Join-Path $stage "models\recon"), (Join-Path $stage "LICENSES") | Out-Null

# 3. Executable.
Copy-Item $exe $stage

# 4. Runtime weights.
#
# DINO ships in whichever format the variant's backend actually loads:
#   gpu/wgpu -> dino_weights.safetensors (BURN runs DINO on the GPU)
#   cpu      -> dino.onnx + dino.onnx.data (use_ort() is true on a CPU build)
# This is a SWAP, not an addition: 326.8 MB of safetensors out, 1.4 + 326.8 MB
# of onnx in, so the zip grows by ~1 MB. app.rs's default-model discovery already
# prefers dino.onnx and falls back to the safetensors, so nothing else changes.
#
# YOLO stays on safetensors for every variant: its use_ort() defaults to false
# (the exported ONNX has a fixed 640x640 input while the Segmentation tab and
# Field Review run at imgsz 1280), so the BURN weights are what actually loads.
$models = Join-Path $root "models"
$need = @(
    @{ src = "yolo_weights.safetensors"; dst = "models\yolo_weights.safetensors" },
    @{ src = "fewshot_head.json";        dst = "models\fewshot_head.json" },
    @{ src = "recon\gen.mpk";            dst = "models\recon\gen.mpk" },
    @{ src = "detector_meta.json";       dst = "models\detector_meta.json" }
)
if ($Variant -eq "cpu") {
    # models\cpu256\ holds the 256-resolution export (1Help/export_dinov3.py
    # --res 256), matching worker::default_dino_res() on a CPU build. It lives in
    # its own folder rather than as dino_256.onnx because ONNX external data is
    # referenced BY FILENAME from inside the graph: this .onnx says
    # "dino.onnx.data", so the pair has to keep those exact names or ort fails to
    # load the weights. Same names, different folder = no collision with the
    # 512 export in models\.
    $need += @{ src = "cpu256\dino.onnx";      dst = "models\dino.onnx" }
    $need += @{ src = "cpu256\dino.onnx.data"; dst = "models\dino.onnx.data" }
}
else {
    $need += @{ src = "dino_weights.safetensors"; dst = "models\dino_weights.safetensors" }
}
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

# 4a. DirectML.dll -- the one native library that is NOT statically linked.
#
# ort-sys emits `cargo:rustc-link-lib=DirectML` (dynamic) in EVERY build variant,
# alongside `static=onnxruntime`. On Windows 10 1903+ DirectML.dll is an inbox
# component in System32 so the loader finds it and nobody notices. On anything
# older -- Win10 LTSC 2019 (1809) and earlier, which is exactly what an old
# conference laptop may be running -- the import fails, and because release
# builds are `windows_subsystem = "windows"` (see main.rs) the process dies
# SILENTLY, with no console and no error dialog. It reads as "the software does
# not work".
#
# 17 MB against a ~750 MB zip is cheap insurance. The loader prefers the
# application directory over System32 for non-KnownDLLs, so the bundled copy
# wins where one is needed and is harmless where the system already has one.
$dml = Join-Path $targetDir "release\DirectML.dll"
if (Test-Path $dml) {
    Copy-Item $dml $stage
    Write-Host "Bundled DirectML.dll (pre-1903 Windows insurance)." -ForegroundColor Green
}
else {
    Write-Warning "DirectML.dll not found at $dml -- package relies on the system copy (Win10 1903+ only)."
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
