#requires -Version 5.1
<#
  Re-copy ShapeComplexity into vendor/ from your working checkout.

  Lacuna depends on `leaf_complex_rust` by path. It used to point at a SIBLING
  directory (`../ShapeComplexity`), which meant the repo could not be built by
  anyone who did not already have that folder — not a collaborator, not CI, not
  a reviewer following the paper.

  It is a vendored copy rather than a git submodule because the upstream repo
  has uncommitted work — including `src/gui_api.rs`, which Lacuna needs — so a
  submodule would pin a commit that does not compile. Once ShapeComplexity is
  committed and pushed, switching to a submodule is the better answer and this
  script becomes unnecessary.

  Usage: powershell -ExecutionPolicy Bypass -File scripts\sync_vendor.ps1 [-Source <path>]
#>
param(
    [string]$Source = "E:\PhD_TobiMu\02_code\ShapeComplexity"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dest = Join-Path $root "vendor\ShapeComplexity"

if (-not (Test-Path $Source)) { throw "source not found: $Source" }
New-Item -ItemType Directory -Force $dest | Out-Null

# /XD target .git .claude — build output and history do not belong in vendor/.
robocopy $Source $dest /MIR /XD target .git .claude /NFL /NDL /NJH /NJS /NP | Out-Null
# robocopy uses 0-7 for success; anything >= 8 is a real failure.
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
$global:LASTEXITCODE = 0

$n = (Get-ChildItem $dest -Recurse -File | Measure-Object).Count
Write-Host "Synced $n files -> vendor\ShapeComplexity" -ForegroundColor Green
Write-Host "Commit the result so CI and collaborators get the same code." -ForegroundColor Green
