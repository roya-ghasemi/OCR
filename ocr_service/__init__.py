"""Persian OCR service — full-text transcription of any image, CPU only.

Layout
    config.py             settings (env-driven): Tesseract, digit model, thresholds
    transcribe.py         image -> every printed line in reading order (+ numbers)
    letter_fields.py      letter fields cut from the transcript by rules (never generated)
    pipeline.py           transcribe -> fields -> response
    schemas.py            response contract
    api.py                FastAPI: POST /ocr, GET /health, GET /jobs/{id}
    tasks.py              Celery app + task (queued mode)
    digit_reader_v2.py    CNN Persian/Latin digit reader (models/digit_cnn.npz)
    digit_cnn_data.py     ...its training-data generator; digit_cnn_train.py its trainer
    numeric_validator.py, numeric_reconcile.py, digit_reader.py
                          number helpers from the earlier VLM path, kept with their tests
    preprocess.py         crop / deskew / compress helpers for clients that pre-process

No language model is used anywhere (E19, ocr_eval/experiments.md).
"""
__version__ = "1.0.0"
