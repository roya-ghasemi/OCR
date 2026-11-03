"""Research and training code that is not part of the deployed service.

    digit_cnn_data.py     synthetic training data for the CNN digit reader
    digit_cnn_train.py    its trainer (needs torch; requirements-train.txt)
    digit_reader_v1.py    the earlier HOG+SVM digit reader, kept as a comparison baseline
    numeric_validator.py  number extraction/validation helpers from the earlier VLM path
    numeric_reconcile.py  reconciliation of VLM numbers against the digit reader
    preprocess.py         client-side crop/deskew/compress reference implementation

The service (ocr_service/) imports none of these; ocr_eval/ tools and tests do.
"""
