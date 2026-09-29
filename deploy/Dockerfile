# API and worker image — CPU only, no model server.
# Tesseract engine from the distro; the Persian/English traineddata are the same
# tessdata_best files the dev box was measured with (E19), NOT the distro package
# (a different, weaker model). /health reports their sha256 so a mismatch is visible.
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

ENV PYTHONUNBUFFERED=1 OCRS_TESSERACT_CMD=/usr/bin/tesseract OCRS_TESSDATA_DIR=/opt/tessdata
EXPOSE 8000
CMD ["uvicorn", "ocr_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
