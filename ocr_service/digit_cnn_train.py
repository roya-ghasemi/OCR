# -*- coding: utf-8 -*-
r"""Train the v2 glyph CNN on `models/digit_glyphs.npz` and export plain numpy weights.

Runs in the torch venv (`venv`, Python 3.14, CUDA); the service venv has no torch, so
the exported `models/digit_cnn.npz` is consumed by a numpy forward pass in
`digit_reader_v2.py`. Holdout is BY FONT (every 8th font), so the accuracy reported
is on typefaces the model never saw.

Architecture (small on purpose — ~2k glyphs per page at inference, CPU):
    conv3x3(1->16) relu pool2 | conv3x3(16->32) relu pool2 | conv3x3(32->64) relu pool2
    -> flatten(1024) ++ 3 line scalars -> fc 128 relu -> fc 25

    venv\Scripts\python.exe -m ocr_service.digit_cnn_train --epochs 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models" / "digit_glyphs.npz"
OUT = ROOT / "models" / "digit_cnn.npz"


class GlyphNet(nn.Module):
    def __init__(self, n_classes: int = 25):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 4 * 4 + 3, 128)
        self.f2 = nn.Linear(128, n_classes)

    def forward(self, x, s):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = torch.cat([x.flatten(1), s], 1)
        return self.f2(F.relu(self.f1(x)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--all-fonts", action="store_true",
                    help="train the exported model on every font (holdout metrics are then from the by-font run in the log)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.data, allow_pickle=False)
    X = torch.from_numpy(z["X"]).float().div_(255.0).unsqueeze(1)
    S = torch.from_numpy(z["S"]).float()
    y = torch.from_numpy(z["y"].astype(np.int64))
    fonts = z["font"].astype(np.int64)
    classes = [str(c) for c in z["classes"]]
    hold = torch.from_numpy(fonts % 8 == 0)
    tr, te = (torch.ones_like(hold) if a.all_fonts else ~hold), hold
    font_names = [str(f).lower() for f in z["fonts"]]
    OFFICE = ("nazanin", "lotus", "mitra", "zar", "titr", "yekan", "tahoma", "arial", "roya", "koodak", "homa", "badr", "compset", "traffic", "calibri", "times")
    office = torch.from_numpy(np.array([any(k in font_names[f] for k in OFFICE) for f in fonts]))
    te_office = (office[te]).to(dev)
    print(f"samples {len(y)}  train {int(tr.sum())}  holdout(by font) {int(te.sum())}  device {dev}")

    # class weights: the reject class is still the majority; digits matter more
    counts = torch.bincount(y[tr], minlength=len(classes)).float()
    w = (counts.sum() / (len(classes) * counts.clamp(min=1))).sqrt()
    net = GlyphNet(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((int(tr.sum()) + a.batch - 1) // a.batch))
    Xtr, Str, ytr = X[tr].to(dev), S[tr].to(dev), y[tr].to(dev)
    Xte, Ste, yte = X[te].to(dev), S[te].to(dev), y[te].to(dev)
    wt = w.to(dev)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(ytr), device=dev)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb = Xtr[idx]
            # light train-time augmentation: random 1-px shifts
            if torch.rand(1).item() < 0.5:
                dx, dy = int(torch.randint(-1, 2, (1,))), int(torch.randint(-1, 2, (1,)))
                xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            logits = net(xb, Str[idx])
            loss = F.cross_entropy(logits, ytr[idx], weight=wt, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i + 4096], Ste[i:i + 4096]).argmax(1) for i in range(0, len(yte), 4096)])
        acc = (pred == yte).float().mean().item()
        is_digit = yte < 20
        digit_acc = (pred[is_digit] == yte[is_digit]).float().mean().item()
        rej_recall = (pred[yte == 24] == 24).float().mean().item()
        false_digit = ((pred < 20) & (yte == 24)).float().sum().item() / max(int((yte == 24).sum()), 1)
        od = is_digit & te_office
        office_digit_acc = (pred[od] == yte[od]).float().mean().item()
        print(f"epoch {ep + 1:2d} loss {tot / len(ytr):.4f} | holdout acc {acc:.4f} digit acc {digit_acc:.4f} office-font digit acc {office_digit_acc:.4f} "
              f"reject recall {rej_recall:.4f} letter->digit {false_digit:.4f} | {time.perf_counter() - t0:.0f}s", flush=True)

    # confusion among digit values (script-agnostic) on holdout, for the report
    with torch.no_grad():
        pv = pred.clone(); yv = yte.clone()
    conf = {}
    for t in range(20):
        m = yv == t
        if m.any():
            wrong = pv[m][pv[m] != t]
            top = torch.bincount(wrong, minlength=25).topk(2)
            conf[classes[t]] = {"n": int(m.sum()), "acc": round((pv[m] == t).float().mean().item(), 4),
                                "top_confusions": [(classes[int(i)], int(c)) for c, i in zip(top.values, top.indices) if c > 0]}
    net.cpu()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(a.out, **sd, classes=np.array(classes))
    info = {"holdout_acc": round(acc, 4), "holdout_digit_acc": round(digit_acc, 4), "holdout_office_digit_acc": round(office_digit_acc, 4), "reject_recall": round(rej_recall, 4),
            "letter_to_digit_rate": round(false_digit, 4), "epochs": a.epochs, "samples": int(len(y)),
            "holdout_fonts": int(len(set(fonts[hold.numpy()]))), "trained_on_all_fonts": a.all_fonts,
            "per_class": conf, "out": a.out}
    (Path(a.out).with_suffix(".json")).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "per_class"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
