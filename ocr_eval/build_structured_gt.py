"""
Phase 1 deliverable — structured (per-field) ground truth.

The flat-blob scoring in run_eval.py concatenates all five predicted fields and
compares to one flat GT string. That conflates FOUR different things into a single
CER number:
  1. character reading errors            (the thing we actually want to measure)
  2. field misassignment                 (right text, wrong JSON key)
  3. schema-orphaned content             (text on the page the schema can't hold)
  4. omission / hallucination            (missing vs invented content)

To separate them we need to know, for each image, which characters belong to which
field. This script reconstructs that WITHOUT re-rendering the images (so the frozen,
byte-identical dataset is never touched): it re-derives each letter's field
decomposition from the SAME content pools and assembly logic used by
generate_dataset.py, and validates that the pieces re-join to the stored GT text.

The field decomposition follows the model's ACTUAL instructions (main.py
_SYSTEM_PROMPT), not an idealized schema:
  sender       <- letterhead org / signatory
  receiver     <- (these letters have NO addressee -> correct output is null)
  subject      <- subject value
  body_text    <- greeting + body + closing + signature, verbatim
  contact_info <- email / phone ONLY (prompt: address, phone, fax, postal, email)

Everything else the bilingual letters print — date, economic code, contract no.,
amount, national id, IBAN, tracking code, discount, version — has NO home in the
five-field schema. It is tagged `__orphan__`: content that is on the page but that
a perfectly-behaved model has nowhere to put. Charging it as reading error (as the
flat blob does) is a measurement bug, not a model failure.

Output: ocr_eval/ground_truth_fields.jsonl  (one row per image)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_dataset as G  # safe: build() only runs under __main__, import renders nothing

GT_FLAT = HERE / "ground_truth.jsonl"
OUT = HERE / "ground_truth_fields.jsonl"

GREETING = "با سلام و احترام،"
CONTACT_LABELS = ("ایمیل", "شماره تماس")  # the only BI_FIELDS that map to contact_info


def _sig_parts(sig: str):
    """A signature is 'role\\nname'. Return (role, name) as separate segments."""
    parts = sig.split("\n")
    return parts[0], "\n".join(parts[1:]) if len(parts) > 1 else ""


def fa_decomposition(i: int):
    sender = G.SENDERS_FA[i % len(G.SENDERS_FA)]
    subject = G.SUBJECTS_FA[i % len(G.SUBJECTS_FA)]
    body = G.BODIES_FA[i % len(G.BODIES_FA)]
    closing = G.CLOSINGS_FA[i % len(G.CLOSINGS_FA)]
    role, name = _sig_parts(G.SIGNATURES_FA[i % len(G.SIGNATURES_FA)])

    segments = [
        {"text": sender, "belongs": "sender"},
        {"text": subject, "belongs": "subject"},
        {"text": GREETING, "belongs": "body_text"},
        {"text": body, "belongs": "body_text"},
        {"text": closing, "belongs": "body_text"},
        {"text": role, "belongs": "body_text"},
        {"text": name, "belongs": "body_text"},
    ]
    fields = {
        "sender": sender,
        "subject": subject,
        "body_text": "\n".join([GREETING, body, closing, role, name]),
    }
    return fields, segments, []  # no orphans in Persian-only letters


def bi_decomposition(i: int):
    sender = G.SENDERS_BI[i % len(G.SENDERS_BI)]
    subject = G.SUBJECTS_FA[(i + 2) % len(G.SUBJECTS_FA)]
    date = G.DATES[i % len(G.DATES)]
    body = G.BODIES_FA[(i + 1) % len(G.BODIES_FA)]
    f1 = G.BI_FIELDS[(2 * i) % len(G.BI_FIELDS)]
    f2 = G.BI_FIELDS[(2 * i + 1) % len(G.BI_FIELDS)]
    closing = G.CLOSINGS_FA[i % len(G.CLOSINGS_FA)]
    role, name = _sig_parts(G.SIGNATURES_BI[i % len(G.SIGNATURES_BI)])

    segments = [
        {"text": sender, "belongs": "sender"},
        {"text": subject, "belongs": "subject"},
        {"text": f"تاریخ: {date}", "belongs": "__orphan__"},
        {"text": GREETING, "belongs": "body_text"},
        {"text": body, "belongs": "body_text"},
    ]
    contact_vals = []
    orphans = [f"تاریخ: {date}"]
    for f in (f1, f2):
        label = f.split(":")[0].strip()
        if any(label.startswith(c) for c in CONTACT_LABELS):
            segments.append({"text": f, "belongs": "contact_info"})
            contact_vals.append(f)
        else:
            segments.append({"text": f, "belongs": "__orphan__"})
            orphans.append(f)
    segments += [
        {"text": closing, "belongs": "body_text"},
        {"text": role, "belongs": "body_text"},
        {"text": name, "belongs": "body_text"},
    ]

    fields = {
        "sender": sender,
        "subject": subject,
        "body_text": "\n".join([GREETING, body, closing, role, name]),
    }
    if contact_vals:
        fields["contact_info"] = "\n".join(contact_vals)
    return fields, segments, orphans


def main():
    if not GT_FLAT.exists():
        sys.exit("ground_truth.jsonl not found; run generate_dataset.py first.")
    flat = {json.loads(l)["filename"]: json.loads(l)
            for l in GT_FLAT.read_text(encoding="utf-8").splitlines() if l.strip()}

    rows = []
    problems = []
    # Re-walk the exact emit order: 18 fa (idx 1..18), then 18 bi (idx 19..36).
    order = [("synthetic_fa", i, i + 1) for i in range(18)] + \
            [("synthetic_bilingual", i, i + 19) for i in range(18)]

    for source, i, idx in order:
        fname = f"{source}_{idx:03d}.png"
        if source == "synthetic_fa":
            fields, segments, orphans = fa_decomposition(i)
        else:
            fields, segments, orphans = bi_decomposition(i)

        # Validation: every segment's text must be present in the stored flat GT.
        gt_text = flat[fname]["text"]
        gt_norm = gt_text.replace("\n", " ")
        for seg in segments:
            for chunk in seg["text"].split("\n"):
                if chunk.strip() and chunk not in gt_norm and chunk not in gt_text:
                    problems.append(f"{fname}: segment not found in GT: {chunk!r}")

        rows.append({
            "filename": fname,
            "source": source,
            "text": gt_text,
            "fields": fields,          # canonical, schema-aligned (receiver intentionally absent)
            "orphan": orphans,         # on-page content with no schema home
            "segments": segments,      # fine-grained, each tagged with its canonical field
        })

    if problems:
        print("VALIDATION FAILED — decomposition does not match stored GT:")
        for p in problems[:20]:
            print("  " + p)
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_orphan = sum(len(r["orphan"]) for r in rows)
    n_contact = sum(1 for r in rows if "contact_info" in r["fields"])
    print(f"Wrote {OUT.name}: {len(rows)} images, decomposition validated against flat GT.")
    print(f"  images with a real contact_info field : {n_contact}")
    print(f"  schema-orphaned content segments total: {n_orphan} "
          f"(date + economic/contract/amount/IBAN/... lines)")
    print(f"  receiver field: ABSENT in all {len(rows)} letters -> correct output is null")


if __name__ == "__main__":
    main()
