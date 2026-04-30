param(
    [string]$DatasetPath = "data",
    [string]$ColmapExe = "",
    [switch]$UseCpu,
    [switch]$ResetOutputs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$datasetRoot = Join-Path $scriptDir $DatasetPath
$imagePath = Join-Path $datasetRoot "images"
$colmapRoot = Join-Path $datasetRoot "colmap"
$sparsePath = Join-Path $colmapRoot "sparse"
$densePath = Join-Path $colmapRoot "dense"
$databasePath = Join-Path $colmapRoot "database.db"

if ([string]::IsNullOrWhiteSpace($ColmapExe)) {
    if ($env:COLMAP_EXE) {
        $ColmapExe = $env:COLMAP_EXE
    } else {
        $ColmapExe = "colmap"
    }
}

function Resolve-ColmapExecutable {
    param([string]$Candidate)

    $command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    if (Test-Path -LiteralPath $Candidate) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    throw "COLMAP executable not found. Add colmap.exe to PATH, set COLMAP_EXE, or pass -ColmapExe."
}

function Get-ColmapVersionString {
    try {
        $helpText = & $script:ResolvedColmapExe -h 2>&1 | Select-Object -First 1
        return [string]$helpText
    } catch {
        return ""
    }
}

function Get-NvidiaGpuNames {
    try {
        $gpuLines = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }
        return @($gpuLines | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    } catch {
        return @()
    }
}

function Invoke-ColmapStep {
    param(
        [string]$Title,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    & $script:ResolvedColmapExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP step failed: $Title"
    }
}

if (-not (Test-Path -LiteralPath $imagePath)) {
    throw "Image directory not found: $imagePath"
}

$script:ResolvedColmapExe = Resolve-ColmapExecutable -Candidate $ColmapExe
$colmapVersionLine = Get-ColmapVersionString
$gpuNames = Get-NvidiaGpuNames

if ($colmapVersionLine -match 'COLMAP 4\.0\.2' -and ($gpuNames | Where-Object { $_ -match 'RTX 50|50.. Ti|5070|5080|5090|Blackwell' })) {
    Write-Warning "Detected COLMAP 4.0.2 on an RTX 50-series / Blackwell GPU. COLMAP 4.0.3 (released 2026-04-06) fixes empty PatchMatch results on Blackwell GPUs. If dense reconstruction returns zero points, upgrade COLMAP before retrying."
}

if ($ResetOutputs) {
    if ((Test-Path -LiteralPath $colmapRoot) -and ($colmapRoot.StartsWith($datasetRoot))) {
        Remove-Item -LiteralPath $colmapRoot -Recurse -Force
    }
}

New-Item -ItemType Directory -Force -Path $sparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $densePath | Out-Null

$featureArgs = @(
    "feature_extractor",
    "--database_path", $databasePath,
    "--image_path", $imagePath,
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.single_camera", "1"
)
$matchingArgs = @(
    "exhaustive_matcher",
    "--database_path", $databasePath
)

if ($UseCpu) {
    $featureArgs += @("--SiftExtraction.use_gpu", "0")
    $matchingArgs += @("--SiftMatching.use_gpu", "0")
}

Invoke-ColmapStep -Title "Step 1: Feature Extraction" -Arguments $featureArgs
Invoke-ColmapStep -Title "Step 2: Feature Matching" -Arguments $matchingArgs
Invoke-ColmapStep -Title "Step 3: Sparse Reconstruction (Bundle Adjustment)" -Arguments @(
    "mapper",
    "--database_path", $databasePath,
    "--image_path", $imagePath,
    "--output_path", $sparsePath
)
Invoke-ColmapStep -Title "Step 4: Image Undistortion" -Arguments @(
    "image_undistorter",
    "--image_path", $imagePath,
    "--input_path", (Join-Path $sparsePath "0"),
    "--output_path", $densePath
)
Invoke-ColmapStep -Title "Step 5: Dense Reconstruction (Patch Match Stereo)" -Arguments @(
    "patch_match_stereo",
    "--workspace_path", $densePath,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "1",
    "--PatchMatchStereo.filter", "1",
    "--PatchMatchStereo.write_consistency_graph", "1"
)
Invoke-ColmapStep -Title "Step 6: Stereo Fusion" -Arguments @(
    "stereo_fusion",
    "--workspace_path", $densePath,
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", (Join-Path $densePath "fused.ply")
)

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "COLMAP executable: $script:ResolvedColmapExe"
Write-Host "Sparse result: $(Join-Path $sparsePath '0')"
Write-Host "Dense result:  $(Join-Path $densePath 'fused.ply')"
