param(
    [switch]$Fast,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "EduPulse AI" -ForegroundColor DarkRed
Write-Host "Student Success Early-Warning and Intervention Platform" -ForegroundColor DarkYellow
Write-Host "Academic prototype localized for a Kabarak University context" -ForegroundColor DarkGray
Write-Host ""

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$modelPath = Join-Path $PSScriptRoot "artifacts\edupulse_model.joblib"
$resultsPath = Join-Path $PSScriptRoot "artifacts\results.json"
$dataPath = Join-Path $PSScriptRoot "data\raw\student_dropout_success.csv"

if ($Fast) {
    if (!(Test-Path $venvPython) -or !(Test-Path $modelPath) -or !(Test-Path $resultsPath)) {
        Write-Host "Fast start is unavailable because setup artifacts are missing." -ForegroundColor Red
        Write-Host "Run START_EDUPULSE.bat once to complete setup." -ForegroundColor Yellow
        exit 1
    }
} else {
    if (!(Test-Path $venvPython)) {
        Write-Host "[1/4] Creating the isolated Python environment..." -ForegroundColor Cyan
        python -m venv .venv
    } else {
        Write-Host "[1/4] Python environment ready." -ForegroundColor Green
    }

    $requirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath "requirements.txt").Hash
    $stampPath = Join-Path $PSScriptRoot ".venv\edupulse-requirements.sha256"
    $installedHash = if (Test-Path $stampPath) { (Get-Content -LiteralPath $stampPath -Raw).Trim() } else { "" }
    if ($installedHash -ne $requirementsHash) {
        Write-Host "      Installing or updating required packages..." -ForegroundColor Cyan
        & $venvPython -m pip install -r requirements.txt
        Set-Content -LiteralPath $stampPath -Value $requirementsHash -Encoding ascii
    }

    if (!(Test-Path $dataPath)) {
        Write-Host "[2/4] Fetching the official UCI Dataset 697..." -ForegroundColor Cyan
        & $venvPython -m scripts.fetch_data
    } else {
        Write-Host "[2/4] Immutable raw dataset ready." -ForegroundColor Green
    }

    $trainingSources = @(
        "src\config.py", "src\data.py", "src\features.py", "src\modeling.py",
        "src\explainability.py", "scripts\train.py", "requirements.txt"
    ) | ForEach-Object { Join-Path $PSScriptRoot $_ }
    $needsTraining = $Retrain -or !(Test-Path $modelPath) -or !(Test-Path $resultsPath)
    if (!$needsTraining) {
        $artifactTime = (Get-Item -LiteralPath $modelPath).LastWriteTimeUtc
        $needsTraining = ($trainingSources | Where-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc -gt $artifactTime }).Count -gt 0
    }
    if ($needsTraining) {
        Write-Host "[3/4] Training and validating the model (training data only for selection)..." -ForegroundColor Cyan
        & $venvPython -m scripts.train
    } else {
        Write-Host "[3/4] Current trained model ready; retraining skipped." -ForegroundColor Green
    }
}

Write-Host "[4/4] Starting the local portal at http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the portal." -ForegroundColor DarkGray
& $venvPython -m streamlit run app.py --server.address localhost --server.port 8501
