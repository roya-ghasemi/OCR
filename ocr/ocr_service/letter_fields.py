# -*- coding: utf-8 -*-
"""Administrative-letter fields cut out of the transcript by rules.

Every value is a verbatim run of transcribed lines, or null. Nothing is generated,
summarised or inferred: a field that is not printed on the page is null. In
particular `subject` is only the text after a printed «موضوع:» label - the corpus
GT carries annotator-written subject summaries for letters that print none, and
producing those is exactly the invention this service no longer does.

A page with neither a salutation/opening formula nor an addressee line is not
treated as a letter and gets all-null fields (its content is in `text`).
"""
from __future__ import annotations

import re
from typing import Sequence

# Lines can start with a stray 1-2 character token the transcriber could not rule out
# (a margin mark next to the text), so line-start patterns tolerate one.
_LEAD = r"^(?:\S{1,2}\s+)?\W*"
_SALUTATION = re.compile(_LEAD + r"(با\s*سلام|باسلام|سلام\s*علیکم|سلام\s*و\s*احترام|احتراما|احتراماً)"
                         # a line that is little more than «سلام» («یاسلام», «پاسلام»: typo or misread «با»)
                         r"|^\W*\S{0,2}\s*سلام\W*$")
_ADDRESSEE = re.compile(r"(جناب|سرکار|ریاست|محترم|حضور|" + _LEAD + r"به\s*[:：])")
_SUBJECT = re.compile(_LEAD + r"موضوع\s*[:：]\s*(.+)$")
_CONTACT = re.compile(r"(آدرس|نشانی|تلفن|تلفکس|فکس|نمابر|کد\s*پستی|صندوق\s*پستی|ایمیل|پست\s*الکترونیک|"
                      r"e-?mail|www\.|https?://|@|وب\s*سایت)", re.I)
_HEADER = re.compile(_LEAD + r"(شماره|تاریخ|پیوست|بسمه|باسمه|به\s*نام\s*خدا|بسم)"
                     r"|(شماره|تاریخ|پیوست)\s*[:：]|بسمه|باسمه|به\s*نام\s*(خدا|او|آن|حق)")
MIN_RECEIVER_CONF = 0.5
_ORG = re.compile(r"(شرکت|سازمان|اداره|وزارت|دانشگاه|بانک|موسسه|مؤسسه|گروه|شهرداری|دفتر|هلدینگ|انجمن|کانون|مجتمع)")

FIELDS = ("sender", "receiver", "subject", "body_text", "contact_info")
_NONWORD = re.compile(r"[^\w‌]")


def _is_salutation(t: str) -> bool:
    """The opening formula, tolerant of OCR damage around it: exact «سلام» /
    «احتراما» among the first three words («اشه احترام بر مذاکرات» for «احتراما پیرو
    مذاکرات»), but never the closing «با احترام»."""
    if _SALUTATION.search(t):
        return True
    toks = [_NONWORD.sub("", w) for w in t.split()[:3]]
    for i, k in enumerate(toks):
        if k in ("سلام", "باسلام", "احتراما", "احتراماً"):
            return True
        if k == "احترام" and not (i > 0 and toks[i - 1] == "با"):
            return True
    return False


def extract(lines: Sequence, page_height: int) -> tuple[dict, bool]:
    """(fields, is_letter). `lines` are transcript lines in reading order with
    `.text` and `.bbox` (x0, y0, x1, y1)."""
    fields: dict = {f: None for f in FIELDS}
    texts = [L.text.strip() for L in lines]
    n = len(texts)
    if not n:
        return fields, False

    sal = next((i for i, t in enumerate(texts) if _is_salutation(t)), None)
    # contact block: contact-looking lines in the bottom third, contiguous to the end
    bottom = [i for i, L in enumerate(lines) if L.bbox[1] > 0.66 * page_height]
    contact_idx = [i for i in bottom if _CONTACT.search(texts[i])]
    c_start = None
    if contact_idx:
        c_start = contact_idx[0]
        fields["contact_info"] = "\n".join(texts[i] for i in range(c_start, n) if i in contact_idx
                                           or (i > c_start and re.search(r"\d{4,}|@", texts[i])))

    # addressee: walking up from the salutation, skipping a printed subject line, the
    # nearest confident line; a second line above it only if it reads as an addressee
    # («جناب ...» over «معاون محترم ...»). Form labels / «به نام ...» end the block.
    head_end = sal if sal is not None else min(n, 6)
    recv = []
    for i in range(head_end - 1, -1, -1):
        t = texts[i]
        if _SUBJECT.search(t):
            continue
        if _HEADER.search(t) or len(recv) == 2:
            break
        conf = getattr(lines[i], "confidence", 1.0)
        if not recv:
            if conf >= MIN_RECEIVER_CONF and (sal is not None or _ADDRESSEE.search(t)):
                recv.append(i)
            elif sal is None:
                continue
            else:
                break
        elif _ADDRESSEE.search(t) and conf >= MIN_RECEIVER_CONF:
            recv.insert(0, i)
        else:
            break
    if recv:
        fields["receiver"] = "\n".join(texts[i] for i in recv)

    subj = next((m.group(1).strip() for t in texts if (m := _SUBJECT.search(t))), None)
    fields["subject"] = subj or None

    if sal is not None:
        end = c_start if c_start is not None and c_start > sal else n
        body = [t for i, t in enumerate(texts[sal:end], sal) if not _SUBJECT.search(t)]
        fields["body_text"] = "\n".join(body) or None

    # sender: an organisation name printed above the addressee (letterhead)
    top = recv[0] if recv else head_end
    for i in range(0, top):
        if _ORG.search(texts[i]) and not _HEADER.search(texts[i]) and not _ADDRESSEE.search(texts[i]):
            fields["sender"] = texts[i]
            break

    is_letter = sal is not None or fields["receiver"] is not None
    if not is_letter:
        return {f: None for f in FIELDS}, False
    return fields, True
