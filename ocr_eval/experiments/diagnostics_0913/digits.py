# -*- coding: utf-8 -*-
r"""E13a — is the image reaching the model, and can it read digits at all?

Synthetic "digit cards": white page, one line of text, Tahoma, rendered at a
known digit height. The engine is called DIRECTLY (/v1/chat/completions on
llama-server) with a plain transcription prompt, so the /ocr JSON prompt, the
grammar and the sampler profiles are all out of the loop. Greedy decoding.

Controls:
  blank      a white page. If the model produces a letter, it is generating
             from the prompt, not reading.
  latin/fa   the same number in Latin and in Persian-Indic digits, at three
             digit heights. If Latin reads and Persian does not, the tokenizer
             (byte-fallback for ۰-۹) is implicated, not the vision path.

    venv312\Scripts\python.exe ocr_eval/experiments/diagnostics_0913/digits.py
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ENGINE = "http://127.0.0.1:18234"
FONT = r"C:\Windows\Fonts\tahoma.ttf"
FA = "۰۱۲۳۴۵۶۷۸۹"
TO_FA = str.maketrans("0123456789", FA)
TO_EN = str.maketrans(FA + "٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

PROMPT = ("Transcribe every character in this image exactly as printed. "
          "Output only the transcription, nothing else.")


def card(text: str, px: int, w: int = 2000, h: int = 500) -> Image.Image:
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, px)
    # PIL has no bidi shaping; digits and Latin render fine, Persian words are
    # rendered with libraqm when available. Keep the Persian to labels only.
    try:
        d.text((w - 80, h // 2), text, font=f, fill="black", anchor="rm",
               direction="rtl", features=["-liga"])
    except Exception:
        d.text((w - 80, h // 2), text, font=f, fill="black", anchor="rm")
    return im


def ask(im: Image.Image, prompt: str = PROMPT, grammar: str | None = None) -> tuple[str, float]:
    buf = io.BytesIO(); im.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt}]}],
            "temperature": 0.0, "top_k": 1, "max_tokens": 200, "cache_prompt": False}
    if grammar:
        body["grammar"] = grammar
    t0 = time.perf_counter()
    with httpx.Client(timeout=300) as c:
        r = c.post(f"{ENGINE}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], round(time.perf_counter() - t0, 2)


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s.translate(TO_EN))


def main() -> int:
    random.seed(13)
    out_dir = HERE / "cards"; out_dir.mkdir(exist_ok=True)
    results = []

    # --- control: blank page -----------------------------------------------
    txt, secs = ask(Image.new("RGB", (2000, 500), "white"))
    results.append({"case": "blank_page", "expected": "", "got": txt, "seconds": secs})
    print(f"blank page        -> {txt[:120]!r}  ({secs}s)")

    # --- digit cards -------------------------------------------------------
    numbers = ["321000000", "0924424117", str(random.randint(10**9, 10**10 - 1)),
               str(random.randint(10**9, 10**10 - 1))]
    for px in (40, 80, 160):
        for n in numbers:
            for script, s in (("latin", n), ("persian", n.translate(TO_FA))):
                im = card(s, px)
                im.save(out_dir / f"{script}_{n}_{px}px.png")
                txt, secs = ask(im)
                got = digits_only(txt)
                ok = got == n
                results.append({"case": f"{script}_{px}px", "expected": n, "raw": txt,
                                "got_digits": got, "exact": ok, "seconds": secs})
                print(f"{script:7s} {px:3d}px  exp={n}  got={got:<14s} {'OK ' if ok else 'BAD'}  raw={txt[:40]!r}")

    # --- mixed line and a labelled amount ------------------------------------
    for label, s, exp in (("mixed_line", "شماره: 1403/4/۵۲۴", "14034524"),
                          ("amount_fa", "مبلغ: ۳۲۱/۰۰۰/۰۰۰ ریال", "321000000"),
                          ("amount_en", "Amount: 321,000,000 Rials", "321000000")):
        im = card(s, 80); im.save(out_dir / f"{label}.png")
        txt, secs = ask(im)
        got = digits_only(txt)
        results.append({"case": label, "expected": exp, "raw": txt, "got_digits": got,
                        "exact": got == exp, "seconds": secs})
        print(f"{label:11s}      exp={exp}  got={got:<14s} {'OK ' if got == exp else 'BAD'}  raw={txt[:50]!r}")

    # --- constrained decoding: digits-only grammar on the Persian amount ------
    g = 'root ::= [0-9\\u06F0-\\u06F9\\u0660-\\u0669/,.\\-]+'
    im = card("مبلغ: ۳۲۱/۰۰۰/۰۰۰ ریال", 80)
    for gscript, gram in (("digits_any_script", g), ("digits_latin_only", 'root ::= [0-9/,.\\-]+')):
        try:
            txt, secs = ask(im, "What is the amount printed in this image? Answer with the number only.", gram)
            got = digits_only(txt)
            results.append({"case": f"grammar_{gscript}", "expected": "321000000", "raw": txt,
                            "got_digits": got, "exact": got == "321000000", "seconds": secs})
            print(f"grammar {gscript:18s} got={got:<14s} {'OK ' if got == '321000000' else 'BAD'}  raw={txt[:40]!r}")
        except Exception as exc:
            print(f"grammar {gscript}: {type(exc).__name__}: {str(exc)[:100]}")

    summary = {}
    for r in results:
        if "exact" in r:
            k = r["case"].split("_")[0]
            summary.setdefault(k, [0, 0]); summary[k][1] += 1; summary[k][0] += int(r["exact"])
    (HERE / "results_E13a_digits.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": ENGINE, "prompt": PROMPT, "decoding": "temperature=0, top_k=1, cache_prompt=false",
        "summary_exact": {k: f"{v[0]}/{v[1]}" for k, v in summary.items()}, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nexact-match summary:", {k: f"{v[0]}/{v[1]}" for k, v in summary.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
