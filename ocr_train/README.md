# Fine-tuning the Persian recogniser (E29)

Everything through E28 improved the service *around* the recogniser — cutting the page out
of the photo, erasing stamps, judging columns, repairing dot confusions against a word list.
Two problems cannot be reached that way at all, because they are not reading errors:

| | |
|---|---|
| **«،» U+060C** | not in the base model's alphabet. 717 occurrences in 2,000 benchmark labels. `Transcriber._fix_marks` exists only to restore a character the model has no output class for. |
| **Latin + «@»** | not in the alphabet either, so every line is read twice — once as `fas`, once as `eng` — and arbitrated. The letter's own address comes back as `info@sabzgostar-govareshk ۱۲” ;`. |

Measured over 2,000 benchmark labels: **31.4% contain a character the base model cannot
emit.** This package fine-tunes the model with an extended alphabet so those become reading
problems instead of impossible ones.

## Running it

```powershell
venv312\Scripts\python.exe -m ocr_train all
```

or one step at a time — they are separate because they fail for different reasons and take
very different times:

| step | what it does | roughly |
|---|---|---|
| `fonts` | probe which faces render, and which carry Latin | seconds |
| `synth` | render synthetic lines from the project's fonts | ~10 min |
| `benchmark` | carve real line images out of the benchmark parquet | ~2 min |
| `unicharset` | extend the alphabet, build the starter model | seconds |
| `lstmf` | convert pairs to Tesseract's training format | ~15 min |
| `train` | fine-tune | tens of minutes |
| `pack` | best checkpoint → `fas_custom.traineddata` | seconds |
| `install` | copy it into `tessdata/` beside `fas` | instant |
| `eval` | character error on the held-out fifth | ~1 min |

Then switch the service over with one setting:

```powershell
$env:OCRS_LANG_FA = "fas_custom"
```

`fas.traineddata` is left in place, so an A/B needs no file juggling and rolling back is
changing a string.

## Training data

| source | lines | what it teaches |
|---|---|---|
| `Persian-OCR-Double-Benchmark`, printed split | 5,224 train / 1,372 held out | real Persian print: varied faces, weights, background tints, noise |
| synthetic, from `ocr_service/fonts` | 6,000 | kashida, and the administrative register — letter numbers, Jalali dates, 8-digit landlines, amounts, e-mail, the fixed formulas |

The synthetic half exists because the benchmark teaches nothing about the two things this
service actually fails on. Administrative Persian is justified by **stretching letters**, not
by widening spaces, and the label for a stretched line is always the *unstretched* text — so
the model is taught that a kashida carries no meaning.

## Three things to know before quoting any number from this

**The benchmark is a published benchmark.** Frozen, leakage-controlled, with its own
leaderboard. Training on it makes our score on it incomparable with that leaderboard. It is
split by content hash and a fifth is held out and never trained on, so one honest number
survives — but any figure from that fifth must carry the sentence *"the other four fifths
were trained on"*. It is not a score on the published benchmark.

**The handwriting split is not used.** It is page-level, its labels contain newlines, and a
printed-text recogniser trained on handwriting gets worse at print.

**A falling training loss is not evidence of anything.** Whether the fine-tune shipped an
improvement is decided by `ocr_eval/tools/bench_pages.py` and `bench_fulltext.py` — the
service's own instruments, on documents it has never trained on.

## Things that cost an afternoon

* **Pango needs a fontconfig file on Windows** or `text2image` writes an empty box file and
  exits 0. `config.write_fontconfig()` lays one down.
* **Pillow has no raqm here**, so it cannot shape Persian — it would draw isolated
  letterforms and a model trained on those is worse than no training. Hence text2image.
* **Every B-series font has zero Latin glyphs.** A line containing "Tel" renders with a hole
  where the word should be. Latin lines are routed to dual-script faces (`FONTS_DUAL`).
* **The IPT faces render nothing at all** — Pango lists them, they produce 0 ink. Not in the
  font list for that reason.
* **`lstm.train` wants a `.box` file**, not a bare `.gt.txt`, in this build. `write_line_box`
  produces the line-box form: one entry per character carrying the whole line's rectangle,
  terminated by a tab entry.
* **List files must have LF endings.** Python's default on Windows writes CRLF, every path
  then ends with `\r`, and lstmtraining reports `Deserialize header failed` against a path
  that looks perfectly correct in a terminal.
* **`--old_traineddata` is what makes the alphabet extension work.** Without it there is no
  mapping from the base model's 95 output classes to the new 153, and the Persian the model
  already knows is thrown away.
