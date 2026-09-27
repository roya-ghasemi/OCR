# Installs Tesseract OCR (UB Mannheim build) + Persian traineddata, and points the
# service at it. Run from an elevated PowerShell (right-click -> Run as administrator):
#
#     powershell -ExecutionPolicy Bypass -File deploy\install_tesseract.ps1
#
# Idempotent: re-running only fills in what is missing.
$ErrorActionPreference = "Stop"

$candidates = @(
    "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
    "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe",
    "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
)
$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $exe) {
    Write-Host "Installing Tesseract via winget (UB-Mannheim.TesseractOCR)..."
    winget install --id UB-Mannheim.TesseractOCR -e --silent `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget install failed with exit code $LASTEXITCODE" }
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) { throw "tesseract.exe not found after install; check the winget output" }
}
Write-Host "tesseract: $exe"

# Persian language data. tessdata_best is the most accurate LSTM set; ~14 MB.
$tessdata = Join-Path (Split-Path $exe) "tessdata"
foreach ($lang in @("fas", "eng")) {
    $f = Join-Path $tessdata "$lang.traineddata"
    if (-not (Test-Path $f)) {
        $url = "https://github.com/tesseract-ocr/tessdata_best/raw/main/$lang.traineddata"
        Write-Host "downloading $lang.traineddata ..."
        Invoke-WebRequest -Uri $url -OutFile $f -UseBasicParsing
    }
    Write-Host "  $lang.traineddata OK ($([math]::Round((Get-Item $f).Length/1MB,1)) MB)"
}

# Point the service at it (user-scope env var; picked up by ocr_service.config).
[Environment]::SetEnvironmentVariable("OCRS_TESSERACT_CMD", $exe, "User")
$env:OCRS_TESSERACT_CMD = $exe
& $exe --version | Select-Object -First 1
& $exe --list-langs
Write-Host "`nDone. Verify with:  venv312\Scripts\python.exe -c ""from ocr_service.numeric_validator import TesseractReader as T; r=T(); print(r.available(), r.version)"""
