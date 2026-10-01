# -*- coding: utf-8 -*-
r"""Fine-tuning the Persian recogniser (E29).

Why this exists, in one paragraph. Up to E28 every improvement was made *around* the
recogniser: the page was cut out of the photo, stamps were erased, columns were judged,
dot confusions were repaired against a word list. Those are all compensations for things
the recogniser gets wrong, and two of them cannot be compensated at all — the base
`fas` model's alphabet has **no Persian comma** (`،`, U+060C) and **no Latin letters**, so
`_fix_marks` has been restoring a character the model is structurally unable to emit, and
every bilingual line has to be read twice and arbitrated. Measured on 2,000 benchmark
labels: 31.4% contain a character the model cannot output, and 717 of those are the comma.

This package fine-tunes the model itself, with an extended alphabet.

    ocr_train.corpus       the synthetic text: kashida, the administrative register,
                           dates, amounts, phone numbers, e-mail, Latin
    ocr_train.synth        render it with text2image + the project's own fonts
    ocr_train.benchmark    carve real line images out of Persian-OCR-Double-Benchmark
    ocr_train.unicharset   extend the alphabet and build the starter traineddata
    ocr_train.lstmf        turn image + ground truth into Tesseract's training format
    ocr_train.train        drive lstmtraining, then pack fas_custom.traineddata

Run it with `python -m ocr_train <step>`; `python -m ocr_train all` does the lot.

Three decisions that bound what this can claim, all of them recorded here because they are
easy to lose:

* **The handwriting split is not used.** It is page-level — one image, a label with
  newlines in it — so it cannot drive line training without a segmentation step, and a
  printed-text recogniser trained on handwriting gets worse at print. The brief's targets
  are printed administrative letters.
* **Persian-OCR-Double-Benchmark is a published benchmark**, frozen and leakage-controlled,
  with its own leaderboard. Training on it makes our score on it incomparable with that
  leaderboard. It is used anyway because the user asked for it and it is the only source of
  real Persian line images here — but it is split by content hash and a fifth is held out
  and never trained on, so there is still an honest number. Any figure quoted from the
  held-out fifth must say that the other four fifths were trained on.
* **The service's accuracy is measured on `ocr_eval`, not here.** `bench_pages.py` and
  `bench_fulltext.py` are the instruments that decide whether the fine-tune shipped an
  improvement; a falling training loss is not evidence of anything.
"""
__all__ = ["config", "corpus", "synth", "benchmark", "unicharset", "lstmf", "train"]
