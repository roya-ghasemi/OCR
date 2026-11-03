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

# Language data: the same files the Docker image pins (deploy/Dockerfile) — Persian and
# Arabic from tessdata_best, English from tessdata_fast.
$tessdata = Join-Path (Split-Path $exe) "tessdata"
$sources = @{ "fas" = "tessdata_best"; "ara" = "tessdata_best"; "eng" = "tessdata_fast" }
foreach ($lang in $sources.Keys) {
    $f = Join-Path $tessdata "$lang.traineddata"
    if (-not (Test-Path $f)) {
        $url = "https://github.com/tesseract-ocr/$($sources[$lang])/raw/main/$lang.traineddata"
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
Write-Host "`nDone. Start the service and check GET /health — tesseract.available must be true."
