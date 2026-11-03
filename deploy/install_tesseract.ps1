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

# Language data, into the repository's models/tesseract/ beside the fine-tuned model: the
# same files the Docker image pins (deploy/Dockerfile) — Persian and Arabic from
# tessdata_best, English from tessdata_fast.
$tessdata = Join-Path (Split-Path $PSScriptRoot) "models\tesseract"
New-Item -ItemType Directory -Force $tessdata | Out-Null
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

# Copy the program into the project, so the project carries everything it runs with.
$bundle = Join-Path (Split-Path $PSScriptRoot) "tools\tesseract"
if (-not (Test-Path (Join-Path $bundle "tesseract.exe"))) {
    New-Item -ItemType Directory -Force (Split-Path $bundle) | Out-Null
    Copy-Item -Recurse (Split-Path $exe) $bundle
}
$exe = Join-Path $bundle "tesseract.exe"
& $exe --version | Select-Object -First 1
& $exe --list-langs --tessdata-dir $tessdata
Write-Host "Tesseract is in $bundle; .env points at tools/tesseract/tesseract.exe"
Write-Host "`nDone. Start the service and check GET /health — tesseract.available must be true."
