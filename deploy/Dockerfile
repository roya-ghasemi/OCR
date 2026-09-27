# API and worker image. The models are NOT in this image; workers talk to the
# llama-server services over HTTP. Tesseract + Persian traineddata are installed
# here because the numeric validation layer is mandatory in production.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY ocr_service ./ocr_service
COPY ocr_pipeline ./ocr_pipeline
COPY ocr_eval/normalize.py ./ocr_eval/normalize.py
COPY ocr_eval/harness.py   ./ocr_eval/harness.py

ENV PYTHONUNBUFFERED=1 OCRS_TESSERACT_CMD=/usr/bin/tesseract
EXPOSE 8000
CMD ["uvicorn", "ocr_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
