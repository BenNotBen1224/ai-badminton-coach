$ErrorActionPreference = "Stop"
$project = Split-Path $PSScriptRoot -Parent
Set-Location $project
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run.py"
$failed = @()
foreach ($method in @("original", "crop", "rtmpose", "rtmo", "vitpose", "mediapipe", "botsort", "onnx", "movenet", "posewarper", "sapiens")) {
    $interpreter = $python
    if ($method -eq "movenet") {
        $interpreter = Join-Path $PSScriptRoot ".venv-movenet\Scripts\python.exe"
    }
    & $interpreter $runner $method
    if ($LASTEXITCODE -ne 0) {
        $failed += $method
        Write-Warning "Candidate $method failed. Its error is preserved in results\$method\failure.txt; continuing the independent candidates."
    }
}
if ($failed.Count -gt 0) {
    throw "Candidates requiring attention: $($failed -join ', '). Successful candidates remain cached."
}
& $python (Join-Path $PSScriptRoot "verify_gallery.py")
if ($LASTEXITCODE -ne 0) {
    throw "Gallery validation failed. Do not treat the delivery as complete."
}
