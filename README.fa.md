# راهنمای اجرای سرویس OCR فارسی

این راهنما قدم‌به‌قدم می‌گوید سرویس را چطور نصب و اجرا کنید. توضیح فنی کامل در
[README.md](README.md) و [RUNBOOK.md](RUNBOOK.md) است.

**این سرویس چه می‌کند:** یک عکس (نامهٔ اداری، صفحهٔ کتاب، جدول، فرم یا اسکرین‌شات) می‌گیرد
و همهٔ متن چاپی آن را به ترتیب خواندن برمی‌گرداند. برای نامه‌ها گیرنده، موضوع، متن و
اطلاعات تماس را هم جدا می‌کند. فقط با CPU کار می‌کند و GPU لازم ندارد.

---

## ۱. اجرای سریع (روی همین سیستم)

همه چیز از قبل نصب است. در PowerShell:

```powershell
cd D:\OCR_ghasemi\ocr
```

```powershell
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

وقتی پیام `Application startup complete.` آمد (حدود ۲ ثانیه):

| آدرس | کاربرد |
|---|---|
| http://127.0.0.1:8000/ | صفحهٔ آپلود: عکس را بکشید و رها کنید |
| http://127.0.0.1:8000/docs | فرم آزمایش API |
| http://127.0.0.1:8000/health | وضعیت سرویس؛ باید `"status": "ok"` باشد |

برای خاموش کردن: `Ctrl + C` در همان پنجره.

---

## ۲. نصب روی یک سیستم جدید (ویندوز)

پیش‌نیاز: **Python 3.12** و دسترسی به اینترنت برای بار اول.

**۱. کپی پروژه:** کل پوشهٔ `ocr` را کپی کنید. پوشهٔ `venv312` را کپی نکنید؛ روی هر سیستم
از نو ساخته می‌شود.

**۲. ساخت محیط پایتون و نصب کتابخانه‌ها:**

```powershell
py -3.12 -m venv venv312
```

```powershell
.\venv312\Scripts\python.exe -m pip install -r requirements.txt
```

فقط برای اجرای سرویس (بدون تست و ارزیابی) `requirements-service.txt` کافی است.

**۳. نصب Tesseract و مدل‌ها** (PowerShell با دسترسی Administrator):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_tesseract.ps1
```

این اسکریپت Tesseract را نصب می‌کند، آن را در `tools\tesseract` کپی می‌کند، و مدل‌های
`fas`، `eng` و `ara` را در `models\tesseract` دانلود می‌کند. مدل فاین‌تیون‌شدهٔ
`fas_custom` و خوانندهٔ ارقام (`models\digit_cnn.npz`) همراه پروژه هستند.

**۴. فایل تنظیمات:**

```powershell
Copy-Item .env.example .env
```

**۵. اجرا:** مثل بخش ۱.

---

## ۳. فایل تنظیمات `.env`

همهٔ تنظیمات در فایل `D:\OCR_ghasemi\ocr\.env` هستند. مسیرهای نسبی نسبت به پوشهٔ پروژه
خوانده می‌شوند، پس پروژه را هر جا کپی کنید کار می‌کند.

| تنظیم | مقدار فعلی | توضیح |
|---|---|---|
| `OCRS_TESSERACT_CMD` | `tools/tesseract/tesseract.exe` | برنامهٔ Tesseract |
| `OCRS_TESSDATA_DIR` | `models/tesseract` | مدل‌های Tesseract |
| `OCRS_DIGIT_MODEL` | `models/digit_cnn.npz` | خوانندهٔ ارقام فارسی |
| `OCRS_LANG_FA` | `fas_custom` | مدل فارسی؛ `fas` مدل اصلی Tesseract است |
| `OCRS_WORKERS` | `8` | تعداد پروسهٔ هم‌زمان Tesseract برای هر صفحه |
| `OCRS_MAX_CONCURRENT_PAGES` | `2` | چند صفحه هم‌زمان خوانده شود |
| `OCRS_MAX_UPLOAD_MB` | `25` | حداکثر حجم عکس |
| `OCRS_API_KEY` | خالی | اگر مقدار بدهید، درخواست‌ها هدر `X-API-Key` می‌خواهند |
| `OCRS_ASYNC_MODE` | `0` | `1` = حالت صف (بخش ۵) |

بعد از تغییر `.env` سرویس را یک بار خاموش و روشن کنید.

**نکته:** متغیر محیطی ویندوز بر `.env` مقدم است. اگر در تنظیمات ویندوز متغیری با نام
`OCRS_...` تعریف شده باشد، همان استفاده می‌شود. روی این سیستم `OCRS_TESSERACT_CMD` در
تنظیمات کاربری ویندوز تعریف شده و به نسخهٔ `C:\Program Files\Tesseract-OCR` اشاره می‌کند؛
خروجی با نسخهٔ داخل پروژه یکی است.

---

## ۴. استفاده از API

**از خط فرمان:**

```powershell
curl.exe -F "file=@exampel_paper.png" http://127.0.0.1:8000/ocr
```

**از پایتون:**

```python
import httpx

with open("letter.jpg", "rb") as f:
    r = httpx.post("http://127.0.0.1:8000/ocr", files={"file": f}, timeout=120)
d = r.json()
print(d["text"])            # کل متن صفحه
print(d["fields"])          # فیلدهای نامه (فقط اگر نامه باشد)
print(d["numbers"])         # اعداد، با منبع و اعتبار گروه‌های سه‌رقمی
print(d["review_reasons"])  # دلایلی که باید یک نفر چک کند
```

اگر `OCRS_API_KEY` را تنظیم کرده‌اید، هدر `{"X-API-Key": "..."}` را هم بفرستید.

**مهم‌ترین بخش‌های پاسخ:**

| فیلد | معنی |
|---|---|
| `text` | کل متن صفحه به ترتیب خواندن |
| `lines` | هر خط با جای آن روی عکس و میزان اطمینان |
| `numbers` | هر عدد؛ `source: glyph` یعنی خوانندهٔ ارقام تأییدش کرده، `grouping: broken` یعنی یک رقم گم شده و به عدد اعتماد نکنید |
| `fields` | فرستنده، گیرنده، موضوع، متن و تماس (فقط برای نامه) |
| `needs_review` و `review_reasons` | آیا و چرا باید یک نفر عکس را چک کند |
| `unread` | جاهایی که خوانده نشد، مثل دست‌خط |

---

## ۵. حالت صف (برای تعداد زیاد درخواست)

در این حالت `/ocr` فوراً یک `job_id` برمی‌گرداند و نتیجه از `/jobs/{job_id}` گرفته می‌شود.
پردازش را یک یا چند worker جداگانه انجام می‌دهند.

**روی ویندوز بدون Redis:** در `.env` این دو خط را تغییر دهید:

```
OCRS_ASYNC_MODE=1
OCRS_REDIS_URL=filesystem://.celery_fs
```

بعد در **دو پنجرهٔ جدا** از پوشهٔ پروژه اجرا کنید. پنجرهٔ اول، worker:

```powershell
.\venv312\Scripts\python.exe -m celery -A ocr_service.tasks worker -l info
```

پنجرهٔ دوم، API:

```powershell
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

برای سرعت بیشتر، worker را در پنجره‌های بیشتری هم اجرا کنید. روی ویندوز سرویس خودش حالت
درست worker را انتخاب می‌کند و لازم نیست `--pool` بدهید.

**با Redis:** Redis را روشن کنید (`deploy\run_redis_local.ps1`) و در `.env` بگذارید
`OCRS_REDIS_URL=redis://127.0.0.1:6379/0`.

---

## ۶. اجرا روی سرور لینوکس با Docker

```bash
cp .env.example .env
```

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

سرویس روی پورت **8080** باز می‌شود (nginx → API → Redis → worker). برای worker بیشتر:

```bash
docker compose -f deploy/docker-compose.yml up -d --scale worker=3
```

بررسی سلامت:

```bash
curl -s http://localhost:8080/health
```

اجرا با systemd بدون Docker در [RUNBOOK.md](RUNBOOK.md) بخش ۵ آمده است.

---

## ۷. تست و ارزیابی

```powershell
.\venv312\Scripts\python.exe -m pytest
```

```powershell
.\venv312\Scripts\python.exe -m ruff check .
```

اندازه‌گیری دقت روی ۵۱ نامهٔ واقعی (حدود ۳ دقیقه):

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev
```

---

## ۸. رفع مشکلات رایج

| مشکل | راه حل |
|---|---|
| `WinError 10048` هنگام اجرا | پورت 8000 اشغال است؛ سرویس قبلی را ببندید یا `--port 8001` بدهید |
| `/health` می‌گوید `down` | Tesseract یا مدل فارسی پیدا نشد؛ مسیرهای `.env` و پوشهٔ `models\tesseract` را چک کنید |
| `/health` می‌گوید `degraded` | خوانندهٔ ارقام (`models\digit_cnn.npz`) پیدا نشد؛ دقت اعداد پایین می‌آید |
| تغییر `.env` اثر نکرد | سرویس را دوباره اجرا کنید، و چک کنید متغیر محیطی ویندوز با همان نام وجود نداشته باشد |
| در حالت صف jobها در `queued` می‌مانند | worker روشن نیست، یا API و worker مقدار `OCRS_REDIS_URL` متفاوتی دارند |
| خطای `401` | `OCRS_API_KEY` تنظیم شده؛ هدر `X-API-Key` را بفرستید |
| متن پر از غلط نقطه‌ای (ب/پ/ت) | عکس کم‌کیفیت است؛ پیام `image resolution too low` را ببینید و با ۳۰۰ DPI اسکن کنید |
