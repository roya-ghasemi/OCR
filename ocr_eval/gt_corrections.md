# Ground-truth corrections (append-only)

Every edit to `ground_truth.jsonl`, with the reason and the baselines re-run afterwards.
Per operating rule 4, ground truth is never edited to improve a metric.

## Phase 0

**No corrections applied.** The audit in `gt_audit.md` found 18 of 36 images carry a
ground-truth defect, but the cause is in `generate_dataset.py` (defects G1 and G2), not in
the transcription. Editing the ground truth would paper over a generator bug. The fix is
to regenerate the dataset with a correct bidi implementation and a reshaper that preserves
harakat, which is Phase 5 work.
