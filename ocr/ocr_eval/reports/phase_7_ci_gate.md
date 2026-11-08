# Phase 7 — productionisation: the regression gate

Phase 7 asks for the things that keep a fix fixed. The accuracy-based exit
criteria are blocked (**D35**), so this report covers what is not: a gate that
runs on every commit, and the two wiring gaps that stand between the tested code
and the code serving traffic.

## The gate

`ocr_eval/ci_gate.py`. Ground-truth-free by design — a gate that cannot run
without GT is a gate that never runs.

```bash
venv312\Scripts\python.exe ocr_eval/ci_gate.py            # enforce
venv312\Scripts\python.exe ocr_eval/ci_gate.py --offline  # no engine needed
venv312\Scripts\python.exe ocr_eval/ci_gate.py --update-baseline
```

Exit 0 pass, 1 regression, 2 could not run.

### What it checks, and which defect each check would have caught

| check | direction | catches |
|---|---|---|
| `lock:manifest`, `lock:splits_v2` | must match | a silent corpus or split change, which invalidates every historical comparison in this repo |
| `tests:*` — 5 suites | must be green | unit regressions across `ocr_pipeline`, the normalizer, the harness, JSON repair, and the Phase 0 guards |
| `live:usable_output_rate` | floor | D3 — unparseable JSON returning as a failure |
| `live:mean_fields_filled` | floor | **D41** — the `sender`-only stub that a usable-output rate alone cannot see |
| `live:degeneracy_rate` | ceiling | D23, D33, **D39** — repetition loops, including the grammar-induced kind |
| `live:all_null_200_rate` | ceiling | D4 — a silently empty success |
| `live:field_collapse_rate` | ceiling | D31, **D44** — the whole letter written into a header field |
| `live:arabic_letterform_rate` | ceiling | D8, **D43** — Arabic yeh/kaf reaching the caller |
| `live:latency_p95_s` | ceiling | the latency regression that always accompanied a loop |
| `live:transport_errors` | must be 0 | the engine being down while the gate reports green |

`mean_fields_filled` exists because of D41. A gate that watched only
usable-output rate would have called the current state of this service healthy
while 80% of documents came back without a body.

### Design decisions worth stating

**The live sample is drawn from the dev split only.** The test split is scored
once per phase by hand; CI must never consume it, or the held-out number stops
being held out. `dev_sample()` strides across the split rather than taking the
first *n*, so the sample is not dominated by one scanning session.

**Every threshold carries explicit slack**, recorded in `DIRECTION`. On a
12-document sample a rate moves 8.3 points when a single document changes
verdict; without slack the gate would fail on noise and be switched off within a
week. `transport_errors` is the one check with zero slack — the engine is either
reachable or the run is meaningless.

**A metric with no recorded baseline is reported but not enforced**, and says so
in its own output line. A gate that invents a threshold is worse than one that
admits it has none.

**`--update-baseline` is deliberately a separate invocation.** Its file carries
the note *"re-record deliberately, never to make a red gate green."*

## Current state

Offline run, 2026-09-03, all suites green:

```
[PASS] lock:manifest              1 (floor 1)  hash matches
[PASS] lock:splits_v2             1 (floor 1)  hash matches
[PASS] tests:test_ocr_pipeline    1 (floor 1)  green
[PASS] tests:test_normalize       1 (floor 1)  green
[PASS] tests:test_harness         1 (floor 1)  green
[PASS] tests:test_phase0_guards   1 (floor 1)  green
[PASS] tests:test_json_repair     1 (floor 1)  green

7/7 checks passed.
```

## The wiring gap — the real Phase 7 finding

Two defects are **fixed in tested code that is not the code serving traffic**:

- **D43** — Arabic letterforms reach the caller in all five fields.
  `ocr_pipeline/persian_text.py` folds them and is unit-tested.
- **D44** — 20.2% schema collapse, `sender` values up to 2116 characters.
  `ocr_pipeline/grammar.py` makes this unrepresentable with bounded field caps.

`main.py` runs its own inline pipeline and does not import `ocr_pipeline`. Both
are wiring tasks, not research tasks.

Mounting `ocr_pipeline` changes the response contract and the decoding strategy
on the production endpoint. That is a deliberate decision with a rollback story,
not a change to be made silently in the middle of a remediation run, so it is
recorded here rather than performed. The A/B evidence for whether it should
happen is in `phase_6_decoding_ab.md`.

## Exit criteria

| criterion | status |
|---|---|
| regression gate exists and runs without GT | **met** — `ci_gate.py`, 7/7 offline |
| gate covers every closed reliability defect | **met** — table above |
| per-field confidence available for routing | **met** — `ocr_pipeline/validation.py::per_field_confidence` |
| runbook | **met** — `RUNBOOK.md` |
| accuracy regression thresholds | **blocked (D35)** — needs GT |
| fixes wired into the serving path | **not met** — D43, D44 |
