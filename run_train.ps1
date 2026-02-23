param(
    [string]$Workspace = $env:ROBOFLOW_WORKSPACE,
    [string]$Project = $env:ROBOFLOW_PROJECT,
    [int]$Version = $(if ($env:ROBOFLOW_VERSION) { [int]$env:ROBOFLOW_VERSION } else { 0 }),
    [string]$DatasetDir = $env:DATASET_DIR,
    [int]$ImgSize = 224,
    [int]$BatchSize = 32,
    [int]$EpochsHead = 5,
    [int]$EpochsFinetune = 10,
    [int]$FinetuneLayers = 30,
    [string]$OutputDir = "artifacts"
)

$ErrorActionPreference = "Stop"

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*#" -or $_ -match "^\s*$") { return }
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim().Trim('"')
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Set-Item -Path ("Env:" + $name) -Value $value
        }
    }

    if (-not $Workspace) { $Workspace = $env:ROBOFLOW_WORKSPACE }
    if (-not $Project) { $Project = $env:ROBOFLOW_PROJECT }
    if ($Version -eq 0 -and $env:ROBOFLOW_VERSION) { $Version = [int]$env:ROBOFLOW_VERSION }
    if (-not $DatasetDir) { $DatasetDir = $env:DATASET_DIR }
}

if (-not $DatasetDir -and (-not $Workspace -or -not $Project -or $Version -eq 0)) {
    throw "Workspace/Project/Version is required unless DATASET_DIR is set."
}

$cmd = @(
    "python", "train_mobilenet.py",
    "--img-size", "$ImgSize",
    "--batch-size", "$BatchSize",
    "--epochs-head", "$EpochsHead",
    "--epochs-finetune", "$EpochsFinetune",
    "--finetune-layers", "$FinetuneLayers",
    "--output-dir", $OutputDir
)

if ($Workspace) {
    $cmd += @("--workspace", $Workspace)
}
if ($Project) {
    $cmd += @("--project", $Project)
}
if ($Version -ne 0) {
    $cmd += @("--version", "$Version")
}

if ($DatasetDir) {
    $cmd += @("--dataset-dir", $DatasetDir)
}

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length - 1)]
