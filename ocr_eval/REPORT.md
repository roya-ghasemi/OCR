# OCR Accuracy Report

Endpoint tested: `http://127.0.0.1:8000/ocr` (the real `/ocr` product endpoint).  
Model output (5 JSON fields) is flattened into one text blob and scored against ground truth.

## 1. Headline numbers

- **Images:** 36  
- **No output:** 3 hard failures (HTTP 422, unparseable JSON) + 3 empty (200 OK but all fields null) = **16.67%**  
- **Images that produced text:** 30  
- **CER (over images with output):** **17.04%**  
- **WER (over images with output):** **22.38%**  
- **Avg latency:** 3.15 s/image

> There are two distinct failure modes, kept separate from reading accuracy so neither hides the other: (1) **no output** — the model either loops into invalid JSON (422) or returns all-null fields; (2) **reading errors** — measured by CER/WER only on the images where the model actually produced text.

## 2. Accuracy by dataset source

| Source | Scored | Empty | 422 | CER | WER |
|---|--:|--:|--:|--:|--:|
| ALL | 30 | 3 | 3 | 17.04% | 22.38% |
| synthetic_bilingual | 15 | 0 | 3 | 25.59% | 32.44% |
| synthetic_fa | 15 | 3 | 0 | 4.49% | 8.04% |

## 3. Normalization sensitivity (ALL, CER)

Persian CER is famously sensitive to normalization choices, so the same predictions are scored three ways:

| Normalization | CER | WER |
|---|--:|--:|
| raw (whitespace only) | 17.75% | 25.85% |
| + Persian (ی/ك unify, ZWNJ, punctuation) | 17.04% | 22.38% |
| + digit style (۰-۹ → 0-9) | 17.04% | 22.38% |

- Persian normalization changes CER by 0.71% (helps) — mostly ZWNJ and ی/ك unification.  
- Digit-style normalization changes CER by 0.00%. In this set the ground-truth digits are Latin and the model also emits Latin digits, so it has **near-zero effect here** — but it would matter for documents whose numbers are written in Persian digits (۰-۹).

## 4. Bilingual: per-language segment accuracy

On the bilingual letters, Persian-script runs and Latin/digit runs are scored separately (a single blended score would hide which side is weaker):

| Segment | CER |
|---|--:|
| Persian-script | 16.17% |
| Latin / digits | 32.85% |

**The Latin/digit side is the weaker one** (32.85% vs 16.17% CER). Codes, dates, and account numbers embedded in Persian text are read less reliably than the surrounding Persian prose.

## 5. RTL/LTR ordering errors (Latin/digit runs)

- Latin-letter / digit runs checked: **140**  
- Reversed (e.g. a number's digits flipped): **1** (0.71%)  
- Other misreads (wrong characters, not a clean reversal): **45** (32.14%)  
- Reversal examples: `0093->3900`

> Clean full-reversals are rare here, but ~1 in 3 Latin/digit runs is misread in some way — the dominant bilingual failure is character-level misreading of embedded codes/numbers, not whole-token reversal.

## 6. Worst 15 images (highest CER, successful only)

| # | File | CER | Ground truth → Prediction |
|--:|---|--:|---|
| 1 | synthetic_bilingual_024.png | 31.03% | **GT:** اداره فناوری اطلاعات - IT Department ⏎ موضوع: پیگیری قرارداد پشتیبانی سالانه ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ احتراماً به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎ درصد تخفیف: 25% ⏎ نسخه سامانه: version 3.4.1 ⏎ خواهشمند است اقدام مقتضی به عمل آید.  <br> **PR:** IT Department - اطلاعات ⏎ پیگیری قرارداد پشتیبانی سالانه ⏎ موضوع: پیگیری قرارداد پشتیبانی سالانه ⏎ با سلام و احترام، ⏎  ⏎ احتمالا به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎  ⏎ درصد تخفیف: 25% ⏎  ⏎ نسخه سامانه: 3.4.1 ⏎  ⏎ خواهشمند است اقدام مقتضی به عمل آید.  |
| 2 | synthetic_bilingual_036.png | 30.72% | **GT:** اداره فناوری اطلاعات - IT Department ⏎ موضوع: پیگیری قرارداد پشتیبانی سالانه ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ احتراماً به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎ درصد تخفیف: 25% ⏎ نسخه سامانه: version 3.4.1 ⏎ خواهشمند است اقدام مقتضی به عمل آید.  <br> **PR:** IT Department - اطلاعات ⏎ پیگیری قرارداد پشتیبانی سالانه ⏎ موضوع: پیگیری قرارداد پشتیبانی سالانه ⏎ با سلام و احترام، ⏎  ⏎ احتمالاً به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎  ⏎ درصد تخفیف: 25% ⏎ نسخه سامانه: 3.4.1 ⏎  ⏎ خواهشمند است اقدام مقتضی به عمل آید. ⏎  |
| 3 | synthetic_bilingual_028.png | 29.29% | **GT:** اداره فناوری اطلاعات - IT Department ⏎ موضوع: دعوت به جلسه هماهنگی فنی ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: EMP-4821 ⏎ پیشاپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ معاون فنی CTO ⏎ A <br> **PR:** IT Department - اطلاعات ⏎ اداره فناوری اطلاعات - هماهنگی فنی ⏎ موضوع: دعوت به جلسه هماهنگی فنی ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: 4821 ⏎ پیشابیش از همکاری شما سپاسگزاریم. ⏎  ⏎ CTO ⏎ علي رضايی - مع |
| 4 | synthetic_bilingual_025.png | 28.08% | **GT:** شرکت دادهپردازان نوین - NovinData Ltd. ⏎ موضوع: اعلام نتایج ارزیابی عملکرد سهماهه ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمایید موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: INV-2026-0093 ⏎ پیشاپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ مد <br> **PR:** NovinData Ltd ⏎ Roya Ghasemi فر ⏎ اعلان نتایج ارزیابی عملکرد سهامه ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمانیه موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: 3090-2026 ⏎ پیشایپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ مدیرعامل ⏎ Roya Ghasemi فر |
| 5 | synthetic_bilingual_020.png | 27.15% | **GT:** اداره فناوری اطلاعات - IT Department ⏎ موضوع: درخواست تمدید مجوز دسترسی ⏎ تاریخ: 2026-08-27 ⏎  ⏎ با سلام و احترام، ⏎ بدینوسیله گزارش عملکرد واحد فنی جهت استحضار و بهرهبرداری تقدیم میگردد. ⏎ تاریخ: 2026-08-27 ⏎ مبلغ کل: 1,250,000 ریال ⏎ از بذل توجه جنابعالی قدردانی میشود. ⏎  ⏎ معاون فنی CTO ⏎ Ali Rez <br> **PR:** IT Department - اطلاعات ⏎ مدیر عامل ⏎ درخواست تمدید مجوز دسترسی ⏎ با سلام و احترام، ⏎ بدينوسيله گزارش عملکرد واحد فنی جهت استحضاار و بهرهبرداری تقدیم میگردد. ⏎ تاریخ: 2026-08-27 ⏎ مبلغ کل: 1,250,000 ریال ⏎ از بذل توجه جنابعالی قدردانی میشود. ⏎  ⏎ CTO ⏎ Ali Rezaei - معاون فنی ⏎  ⏎ [ناخوانا] |
| 6 | synthetic_bilingual_022.png | 27.09% | **GT:** دفتر پروژه - Project Management Office ⏎ موضوع: دعوت به جلسه هماهنگی فنی ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: EMP-4821 ⏎ پیشاپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ معاون فنی CTO ⏎ <br> **PR:** Project Management Office ⏎ دکتر پروزه ⏎ دعوت به جلسه هماهنگی فنی ⏎ موضوع: دعوت به جلسه هماهنگی فنی ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: 4821 ⏎ پیشایش از همکاری شما سپاسگزاریم |
| 7 | synthetic_bilingual_019.png | 26.69% | **GT:** شرکت سپهرداده دیجیتال - Sepehr Digital Data Co. ⏎ موضوع: اعلام نتایج ارزیابی عملکرد سهماهه ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمایید موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: INV-2026-0093 ⏎ پیشاپیش از همکاری شما سپاسگزاریم <br> **PR:** Sepehr Digital Data Co ⏎ Roya Ghasemi فر ⏎ اعلان نتایج ارزیابی عملکرد سهماهه ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمانیه موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: 3900-2026 ⏎ پیشایپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ مدیرعامل ⏎ Roya Ghasemi فر |
| 8 | synthetic_bilingual_026.png | 26.62% | **GT:** دفتر پروژه - Project Management Office ⏎ موضوع: درخواست تمدید مجوز دسترسی ⏎ تاریخ: 2026-08-27 ⏎  ⏎ با سلام و احترام، ⏎ بدینوسیله گزارش عملکرد واحد فنی جهت استحضار و بهرهبرداری تقدیم میگردد. ⏎ تاریخ: 2026-08-27 ⏎ مبلغ کل: 1,250,000 ریال ⏎ از بذل توجه جنابعالی قدردانی میشود. ⏎  ⏎ معاون فنی CTO ⏎ Ali R <br> **PR:** Project Management Office - دفتر پروژه ⏎ دفتر پروژه ⏎ درخواست تمديد مجوز دسترسی ⏎ با سلام و احترام، ⏎ بدينوسيله گزارش عملکرد واحد فني جهت استحضار و بهرهبرداري تقديم ميگردد. ⏎ تاريخ: 2026-08-27 ⏎ مبلغ كل: 1,250,000 ريال ⏎ از بذل توجه جنابعلي قدردانی ميشود. ⏎  ⏎ CTO ⏎ علي رضايی - |
| 9 | synthetic_bilingual_032.png | 26.46% | **GT:** اداره فناوری اطلاعات - IT Department ⏎ موضوع: درخواست تمدید مجوز دسترسی ⏎ تاریخ: 2026-08-27 ⏎  ⏎ با سلام و احترام، ⏎ بدینوسیله گزارش عملکرد واحد فنی جهت استحضار و بهرهبرداری تقدیم میگردد. ⏎ تاریخ: 2026-08-27 ⏎ مبلغ کل: 1,250,000 ریال ⏎ از بذل توجه جنابعالی قدردانی میشود. ⏎  ⏎ معاون فنی CTO ⏎ Ali Rez <br> **PR:** IT Department - اطلاعات ⏎ مدیر عامل ⏎ درخواست تمدید مجوز دسترسی ⏎ با سلام و احترام، ⏎ بدينوسيله گزارش عملکرد واحد فنی جهت استحضاار و بهرهبرداری تقدیم میگردد. ⏎ تاریخ: 2026-08-27 ⏎ مبلغ کل: 1,250,000 ریال ⏎ از بذل توجه جنابعالی قدردانی میشود. ⏎  ⏎ CTO ⏎ علي رضايی |
| 10 | synthetic_bilingual_031.png | 25.15% | **GT:** شرکت سپهرداده دیجیتال - Sepehr Digital Data Co. ⏎ موضوع: اعلام نتایج ارزیابی عملکرد سهماهه ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمایید موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: INV-2026-0093 ⏎ پیشاپیش از همکاری شما سپاسگزاریم <br> **PR:** Sepehr Digital Data Co - شرکت سپهرداده دیجیتال ⏎ اعلان نتایج ارزیابی عملکرد سهماهه ⏎ با سلام و احترام، ⏎ پیرو مکاتبات قبلی، خواهشمند است دستور فرمانیه موضوع در اسرع وقت پیگیری شود. ⏎ کد اقتصادی: 51096365 ⏎ شماره قرارداد: 3090-2026 ⏎ پیشایپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ مدیرعامل ⏎ Roya Ghasemi - ر |
| 11 | synthetic_bilingual_034.png | 23.08% | **GT:** دفتر پروژه - Project Management Office ⏎ موضوع: دعوت به جلسه هماهنگی فنی ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: EMP-4821 ⏎ پیشاپیش از همکاری شما سپاسگزاریم. ⏎  ⏎ معاون فنی CTO ⏎ <br> **PR:** Project Management Office ⏎ CTO علي رضایی ⏎ دفتر پروژه - موضع: دعوت به جلسه هماهنگی فنی ⏎ تاریخ: 1405-06-05 ⏎  ⏎ با سلام و احترام، ⏎ نظر به اهمیت موضوع، حضور نماینده تامالاختیار در جلسه مورد انتظار است. ⏎ شماره تماس: +98 21 8899 1234 ⏎ شماره پرسنلی: 4821 ⏎ پیشایی از همکاری شما سپاسگزاریم. ⏎  ⏎ Ali R |
| 12 | synthetic_bilingual_030.png | 22.12% | **GT:** دفتر پروژه - Project Management Office ⏎ موضوع: پیگیری قرارداد پشتیبانی سالانه ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ احتراماً به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎ درصد تخفیف: 25% ⏎ نسخه سامانه: version 3.4.1 ⏎ خواهشمند است اقدام مقتضی به عمل آید <br> **PR:** Project Management Office ⏎ CTO ⏎ پیگیری قرارداد پشتیبانی سالانه ⏎ با سلام و احترام، ⏎  ⏎ احتمالا به استحضار میرساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم. ⏎  ⏎ درصد تخفیف: 25% ⏎ نسخه سامانه: 3.4.1 ⏎  ⏎ خواهشمند است اقدام مقتضی به عمل آید. ⏎  ⏎ CTO ⏎  ⏎ علي رضايی |
| 13 | synthetic_bilingual_021.png | 20.92% | **GT:** شرکت دادهپردازان نوین - NovinData Ltd. ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ با عنایت به بررسیهای انجامشده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال میشود. ⏎ ایمیل: info@sepehr-digital.co ⏎ کد ملی: 0071234567 ⏎ خواهشمند است اقدام مقتضی به عمل آید. <br> **PR:** NovinData Ltd - شرکت داده پردازان نوین ⏎ گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ تاریخ: 13/07/2026 ⏎  ⏎ با سلام و احترام, ⏎  ⏎ با عنايت به بررسی های انجام شده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال می شود. ⏎  ⏎ ایمیل: info@sepehr-digital.co ⏎  ⏎ کد ملی: |
| 14 | synthetic_bilingual_033.png | 20.92% | **GT:** شرکت دادهپردازان نوین - NovinData Ltd. ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ با عنایت به بررسیهای انجامشده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال میشود. ⏎ ایمیل: info@sepehr-digital.co ⏎ کد ملی: 0071234567 ⏎ خواهشمند است اقدام مقتضی به عمل آید. <br> **PR:** NovinData Ltd - شرکت داده پردازان نوین ⏎ گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ تاریخ: 13/07/2026 ⏎  ⏎ با سلام و احترام، ⏎  ⏎ با عنايت به بررسی های انجام شده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال می شود. ⏎  ⏎ ایمیل: info@sepehr-digital.co ⏎  ⏎ کد ملی: |
| 15 | synthetic_bilingual_027.png | 19.46% | **GT:** شرکت سپهرداده دیجیتال - Sepehr Digital Data Co. ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ تاریخ: 2026/07/13 ⏎  ⏎ با سلام و احترام، ⏎ با عنایت به بررسیهای انجامشده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال میشود. ⏎ ایمیل: info@sepehr-digital.co ⏎ کد ملی: 0071234567 ⏎ خواهشمند است اقدام مقتضی به <br> **PR:** شرکت سپهرداده دیجیتال ⏎ Roya Ghasemi فر ⏎ موضوع: گزارش وضعیت سامانه و اقدامات اصلاحی ⏎ با سلام و احترام، ⏎ با عنايت به بررسی های انجام شده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال می شود. ⏎  ⏎ اميل: info@sepehr-digital.co ⏎ کد ملی: 67123456700 ⏎  ⏎ خواهشمند است اقدام مقتضی به عمل آید. ⏎  ⏎ مدیرعام |

## 7. No-output failures

**HTTP 422 (model looped into unterminated JSON):** the same repetition failure seen on the original sample letter before `IMAGE_MIN_TOKENS` was raised; it still triggers occasionally on denser bilingual content.

| File | Error (truncated) |
|---|---|
| synthetic_bilingual_023.png | HTTP 422: {"detail":"Model returned unparseable JSON: Unterminated string starting at: line 1 column 135 (char 134). Raw: {\"sender\": \"Sepehr Digital Data Co\ |
| synthetic_bilingual_029.png | HTTP 422: {"detail":"Model returned unparseable JSON: Unterminated string starting at: line 1 column 126 (char 125). Raw: {\"sender\": \"NovinData Ltd\", \"rece |
| synthetic_bilingual_035.png | HTTP 422: {"detail":"Model returned unparseable JSON: Unterminated string starting at: line 1 column 135 (char 134). Raw: {\"sender\": \"Sepehr Digital Data Co\ |

**Empty (200 OK, all fields null):** the model returned a valid but empty extraction — it declined to read the letter at all.

| File | Source |
|---|---|
| synthetic_fa_002.png | synthetic_fa |
| synthetic_fa_005.png | synthetic_fa |
| synthetic_fa_010.png | synthetic_fa |

## 8. Character confusion summary

Most frequent character substitutions (ground-truth → predicted), after normalization:

| ref → pred | count |
|---|--:|
| `ی -> ه` | 4 |
| `پ -> ی` | 3 |
| `م -> ن` | 3 |
| `I -> 3` | 3 |
| `0 -> 6` | 3 |
| `ش -> ی` | 2 |
| `پ -> ب` | 2 |
| `م -> R` | 2 |
| `و -> o` | 2 |
| `ض -> y` | 2 |
| `و -> a` | 2 |
| `: -> G` | 2 |
| `و -> د` | 2 |
| `ض -> ی` | 2 |
| `و -> ر` | 2 |
| `: -> ا` | 2 |
| `ل -> ا` | 2 |
| `- -> گ` | 2 |
| `N -> و` | 2 |
| `o -> ض` | 2 |
| `v -> ع` | 2 |
| `i -> ی` | 2 |
| `n -> ت` | 2 |
| `L -> س` | 2 |
| `t -> ا` | 2 |

Commonly-confused Persian pairs actually observed: `ب/پ`.

## 9. Plain-language conclusion

- **Printed Persian prose is the model's strong suit** — CER 4.49% on Persian-only letters, and only 16.17% on the Persian runs inside bilingual letters.  
- **Embedded Latin/digit content is the weak spot** — 32.85% CER on codes, dates, emails and account numbers, with ~32.14% of such runs misread and occasional digit reversal.  
- **Robustness:** 16.67% of letters return no usable text (3 as HTTP 422, 3 as all-null 200). For a production path, catching the 422 / empty response and retrying with a higher token budget would remove most of these.  
- **Normalization:** report digit-normalized CER as the headline, but the raw vs. normalized gap is small here (17.75% → 17.04%), so these numbers are not an artifact of aggressive normalization.

### How to re-run

```bash
# regenerate images (optional)
venv312\Scripts\python.exe ocr_eval\generate_dataset.py
# run inference + score (server must be running)
venv312\Scripts\python.exe ocr_eval\run_eval.py
# rebuild this report from the last run
venv312\Scripts\python.exe ocr_eval\make_report.py
```