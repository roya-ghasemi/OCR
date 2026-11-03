# Persian OCR service — API and worker image (CPU only).
#
# Runtime models, about 24 MB in total:
#   fas_custom  fine-tuned Persian model (default)        3.3 MB  from the repository
#   fas         stock Persian model, rollback option      3.3 MB  tessdata_best
#   eng         Latin runs inside Persian text            4.1 MB  tessdata_fast
#   ara         re-read of pages that look Arabic        12.6 MB  tessdata_best
#   digit_cnn   Persian/Latin digit reader                0.6 MB  from the repository
# These are the exact files the accuracy figures were measured with; every download
# is pinned to an upstream commit and checked against its sha256.
FROM python:3.12-slim

ARG TESSDATA_BEST=e12c65a915945e4c28e237a9b52bc4a8f39a0cec
ARG TESSDATA_FAST=87416418657359cb625c412a48b6e1d6d41c29bd

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; mkdir -p /opt/tessdata; cd /opt/tessdata; \
    curl -fsSL -o fas.traineddata "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_BEST}/fas.traineddata"; \
    curl -fsSL -o ara.traineddata "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_BEST}/ara.traineddata"; \
    curl -fsSL -o eng.traineddata "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/${TESSDATA_FAST}/eng.traineddata"; \
    printf '%s  %s\n' \
        99e420969b5ddd2cb135b416316a7ed417c59c4faf9e0d28941348f6448114df fas.traineddata \
        ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896 ara.traineddata \
        7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2 eng.traineddata \
      | sha256sum -c -

RUN useradd --system --uid 10001 --home /app ocr
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY tessdata/fas_custom.traineddata /opt/tessdata/fas_custom.traineddata
COPY models/digit_cnn.npz ./models/digit_cnn.npz
COPY ocr_service ./ocr_service
RUN echo "d0e5099d3ea227ba83fe71dba28acb5a774644bd9583c524c435f5bdc8bd9adb  /opt/tessdata/fas_custom.traineddata" | sha256sum -c - \
    && mkdir -p /var/cache/ocr /var/spool/ocr && chown ocr:ocr /var/cache/ocr /var/spool/ocr

ENV PYTHONUNBUFFERED=1 \
    OCRS_TESSERACT_CMD=/usr/bin/tesseract \
    OCRS_TESSDATA_DIR=/opt/tessdata \
    OCRS_CACHE_DIR=/var/cache/ocr \
    OCRS_SPOOL_DIR=/var/spool/ocr \
    OMP_THREAD_LIMIT=1

USER ocr
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status != 200)"
CMD ["uvicorn", "ocr_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
