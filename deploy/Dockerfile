# API and worker image — CPU only, no model server.
# Tesseract engine from the distro; the Persian/English traineddata are the same
# tessdata_best files the dev box was measured with (E19), NOT the distro package
# (a different, weaker model). /health reports their sha256 so a mismatch is visible.
#
# Three models end up in /opt/tessdata and they are not interchangeable:
#   fas_custom  the E30 fine-tune and the default — copied from the repository
#   fas         the stock model, kept as the one-string rollback (OCRS_LANG_FA=fas)
#   eng         Latin runs inside Persian text
# Check the first against /health: lang_fa_sha256 must read d0e5099d3ea227ba.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/tessdata \
    && curl -fsSL -o /opt/tessdata/fas.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/fas.traineddata \
    && curl -fsSL -o /opt/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata

WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY ocr_service ./ocr_service
# The CNN digit reader's weights (git-ignored; build from a checkout that has them,
# or train with `python -m ocr_service.digit_cnn_train`).
COPY models/digit_cnn.npz ./models/digit_cnn.npz
# The E30 fine-tune, which is what OCRS_LANG_FA defaults to. It is tracked in the
# repository because it exists nowhere to download it from, and without it here the
# container starts cleanly and then fails every request on a missing language.
COPY tessdata/fas_custom.traineddata /opt/tessdata/fas_custom.traineddata

ENV PYTHONUNBUFFERED=1 OCRS_TESSERACT_CMD=/usr/bin/tesseract OCRS_TESSDATA_DIR=/opt/tessdata
EXPOSE 8000
CMD ["uvicorn", "ocr_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
