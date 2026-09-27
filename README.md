# Persian Business Letter OCR — API Service

A REST API that receives a scanned or photographed Persian business letter (نامه اداری) and returns
its contents as structured JSON: sender, receiver, subject, body text, and contact information.

Built with **FastAPI** + a **local vision-language model** (coreOCR-7B) running on an embedded
**llama.cpp** engine. Everything runs on your own machine — no cloud API, no internet at request time.

**LM Studio does not need to be running.** The service starts its own inference engine.

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [How it works — architecture](#2-how-it-works--architecture)
3. [Project structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Running the server](#6-running-the-server)
7. [Testing with Postman](#7-testing-with-postman)
8. [Testing with curl](#8-testing-with-curl)
9. [API reference](#9-api-reference)
10. [Configuration — environment variables](#10-configuration--environment-variables)
11. [Troubleshooting](#11-troubleshooting)
12. [Accuracy notes and tuning](#12-accuracy-notes-and-tuning)
13. [Change history](#13-change-history)

---

## 1. What this project does

You upload a photo or scan of a Persian business letter and the service returns its content as
structured JSON.

**Example request:** `POST http://localhost:8000/ocr` with the letter image attached.

**Example response:**

```json
{
  "sender": "مدیر محترم محصول",
  "receiver": "مدیر عامل",
  "subject": "درخواست راه اندازی سرویس اینترنت پرو",
  "body_text": "احتراماً به استحضار می‌رساند شرکت سپهرداده دیجیتال ...\n\nمدیرعامل\n\nرویا قاسمی فر",
  "contact_info": null
}
```

Fields that do not appear in the document come back as `null`.

Typical response time on an RTX 5070: **3–5 seconds** per letter.

---

## 2. How it works — architecture

```
   Postman / curl / browser
             │
             │  POST /ocr   (multipart form, image file)
             ▼
   ┌─────────────────────────────┐
   │  FastAPI  (main.py)         │   port 8000
   │  • detects real image type  │
   │  • base64-encodes the image │
   │  • builds the OCR prompt    │
   │  • parses + validates JSON  │
   └──────────────┬──────────────┘
                  │  OpenAI-compatible HTTP call
                  ▼
   ┌─────────────────────────────┐
   │  llama-server  (engine.py)  │   port 18234, started automatically
   │  • coreOCR-7B  Q4_K_S       │
   │  • mmproj vision projector  │
   │  • CUDA 12 on the GPU       │
   └─────────────────────────────┘
```

**The key point:** `engine.py` starts `llama-server.exe` as a child process when the app boots and
shuts it down when the app stops. It finds both the engine binary and the model files on disk by
itself. LM Studio is only the *source* of those files — the LM Studio application is never launched
and does not need to be open.

Startup sequence when you run the server:

1. `engine.py` scans `~/.lmstudio/extensions/backends` and picks the build pinned by
   `ENGINE_BUILD_PIN` (default `2.28.2`), falling back to the newest CUDA 12 backend with a
   loud warning. The pin exists so an LM Studio update cannot silently change the engine
   while `config.ENGINE_BUILD` still claims the old one.
2. It searches `MODEL_SEARCH_ROOTS` in order for the coreOCR GGUF and its matching `mmproj`,
   taking the first root that actually contains a match. Default order is
   `D:\models\models`, then `~/.lmstudio/models`. Models and backends are **separate roots** —
   the models moved off `C:` on 2026-09-05 and the backends did not.
3. It launches `llama-server.exe` on port `18234` and waits until `/v1/models` returns 200.
4. FastAPI finishes starting and begins accepting requests on port 8000.

If something is already listening on port `18234`, that engine is reused instead and is **not**
killed on shutdown.

---

## 3. Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — endpoints, image validation, prompt, JSON parsing |
| `engine.py` | Finds, starts, monitors, and stops the llama.cpp engine |
| `requirements.txt` | Python dependencies |
| `exampel_paper.png` | Sample Persian letter for testing |
| `engine.log` | Engine output — written fresh on every start (useful for debugging) |
| `venv312/` | Working virtual environment (Python 3.12) |
| `main_qwen.py` | Alternative backend: loads Qwen2-VL directly via transformers (not in use) |
| `main_lmstudio_client.py.bak` | Previous version that required LM Studio to be running |
| `test_ocr.py` | Older test suite — **stale**, see [section 11](#11-troubleshooting) |
| `venv/` | Old broken virtual environment (Python 3.14, unusable) — safe to delete |

---

## 4. Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 x64 | Paths below assume Windows |
| Python 3.12 | Installed at `C:\Users\<you>\AppData\Local\Programs\Python\Python312` |
| NVIDIA GPU | Tested on RTX 5070 (12 GB). CPU fallback works but is much slower |
| LM Studio (installed once) | Only to supply the model files and engine binary |

**Files that must exist on disk:**

```
D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF\
    coreOCR-7B-050325-preview.Q4_K_S.gguf          (~4.5 GB)
    coreOCR-7B-050325-preview.mmproj-f16.gguf      (~1.4 GB)

    (previously under C:\Users\<you>\.lmstudio\models\ — either location is found
     automatically; set MODELS_DIR or MODEL_SEARCH_ROOTS for anywhere else)

C:\Users\<you>\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-*\
    llama-server.exe
```

Both the model and the vision projector (`mmproj`) are required. Without the `mmproj` file the model
cannot see images.

---

## 5. Installation

Open PowerShell in the project folder.

### Step 1 — create the virtual environment

```powershell
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe -m venv venv312
```

### Step 2 — install dependencies

```powershell
.\venv312\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

> **Why `--index-url` is needed here.** The machine's `pip.ini` points at the mirror
> `mirror-pypi.runflare.com`, which currently fails with `SSL: WRONG_VERSION_NUMBER`. Passing the
> official index bypasses it. If the mirror starts working again, you can drop the flag.
> The config file is at `C:\Users\<you>\AppData\Roaming\pip\pip.ini`.

### Step 3 — verify

```powershell
.\venv312\Scripts\python.exe -c "import fastapi, uvicorn, openai, PIL; print('deps OK')"
```

### Step 4 — verify the engine and model are found

```powershell
.\venv312\Scripts\python.exe -c "import engine; print(engine.find_backend()); print(engine.find_model())"
```

This prints the paths that will be used. If it raises `FileNotFoundError`, set `MODEL_GGUF` /
`LLAMA_SERVER` explicitly — see [section 10](#10-configuration--environment-variables).

---

## 6. Running the server

```powershell
.\venv312\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Add `--reload` while editing code. Note that every reload restarts the engine and reloads the model.

**What you will see:**

```
INFO:     Started server process [23832]
INFO:     Waiting for application startup.
INFO Starting engine: C:\Users\PC\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2\llama-server.exe
INFO Model:  D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF\coreOCR-7B-050325-preview.Q4_K_S.gguf
INFO mmproj: D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF\coreOCR-7B-050325-preview.mmproj-f16.gguf
INFO Engine log: E:\ghasemi\ocr\engine.log
INFO Engine ready at http://127.0.0.1:18234/v1
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**The server is ready when you see** `Application startup complete.` — roughly 5 seconds after launch.
The startup blocks until the model is fully loaded, so a 200 from `/health` means the model is ready.

Stop with `Ctrl + C`. The engine child process is terminated automatically.

Interactive docs (upload an image straight from the browser): **http://localhost:8000/docs**

---

## 7. Testing with Postman

| Setting | Value |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8000/ocr` |
| Body type | `form-data` |
| Key | `file` — or `image`, both are accepted |
| Key type | **File** (change the dropdown from Text to File) |
| Value | select your letter image |

You do **not** need to set a Content-Type on the row. The service reads the real format from the
file's bytes, so Postman's default `application/octet-stream` is fine.

Click **Send**. You should get `200 OK` with the JSON body.

---

## 8. Testing with curl

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"healthy","model":"...coreOCR-7B-050325-preview.Q4_K_S.gguf","backend":"embedded llama.cpp","mmproj":"...mmproj-f16.gguf"}
```

OCR with the bundled sample letter:

```bash
curl -s -X POST http://127.0.0.1:8000/ocr -F "file=@exampel_paper.png" --max-time 900
```

---

## 9. API reference

### `GET /`

Service metadata. Always available, even while the model is loading.

```json
{
  "service": "Persian Business Letter OCR",
  "version": "4.0.0",
  "model": "C:\\Users\\PC\\.lmstudio\\models\\...\\coreOCR-7B-050325-preview.Q4_K_S.gguf",
  "backend": "embedded llama.cpp"
}
```

### `GET /health`

Returns `200` only when the engine process is alive **and** answering. Returns `503` otherwise.

### `POST /ocr`

Multipart form upload. Accepts a form field named **`file`** or **`image`**.

Accepted image formats: **JPEG, PNG, BMP, TIFF, WEBP** — detected from the file contents, not from
the declared Content-Type.

**Response `200`** — every field is nullable:

| Field | Meaning |
|---|---|
| `sender` | فرستنده — from the letterhead or signature |
| `receiver` | گیرنده |
| `subject` | موضوع |
| `body_text` | متن بدنه, verbatim, including greeting and signature |
| `contact_info` | آدرس، تلفن، فکس، کد پستی، ایمیل |

**Error responses:**

| Code | Meaning |
|---|---|
| `400` | No file sent, empty file, or the bytes are not a readable image |
| `422` | The model returned text that is not valid JSON, or does not match the schema |
| `500` | Inference failed (engine crashed, out of memory, timeout) |
| `503` | Engine not running or not reachable (`/health` only) |

---

## 10. Configuration — environment variables

All optional. Set them in PowerShell before launching, e.g. `$env:CONTEXT_SIZE="32768"`.

### Model and engine selection

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_GGUF` | auto-detected | Full path to the model `.gguf` file |
| `MMPROJ_GGUF` | sibling `*mmproj*.gguf` | Full path to the vision projector |
| `MODEL_GLOB` | `**/*coreOCR*.gguf` | Search pattern used when `MODEL_GGUF` is unset |
| `MODELS_DIR` | first root in `MODEL_SEARCH_ROOTS` holding a match | Where to search for models — overrides the search entirely |
| `MODEL_SEARCH_ROOTS` | `D:\models\models`, then `~/.lmstudio/models` | `os.pathsep`-separated roots tried in order when `MODELS_DIR` is unset |
| `LLAMA_SERVER` | auto-detected | Full path to a `llama-server.exe` of your choice |
| `LMSTUDIO_ROOT` | `~/.lmstudio` | Root used to find **backends** (no longer used to find models) |
| `ENGINE_BUILD_PIN` | `2.28.2` | Backend version to prefer. Set to empty for newest-available |

### Engine runtime

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_HOST` | `127.0.0.1` | Engine bind address |
| `ENGINE_PORT` | `18234` | Engine port |
| `N_GPU_LAYERS` | `99` | Layers offloaded to GPU. Set `0` to force CPU |
| `CONTEXT_SIZE` | `16384` | Context window in tokens |
| `IMAGE_MIN_TOKENS` | `1024` | Minimum image tokens — **do not lower this**, see [section 12](#12-accuracy-notes-and-tuning) |
| `ENGINE_STARTUP_TIMEOUT` | `600` | Seconds to wait for the model to load |
| `ENGINE_LOG` | `engine.log` | Where engine output is written |

### Using an external server instead of the embedded engine

| Variable | Default | Meaning |
|---|---|---|
| `API_BASE_URL` | *(unset)* | If set, no engine is started; requests go here instead |
| `MODEL_NAME` | model file path | Model id sent in the API call |
| `API_KEY` | `local` | Sent as the bearer token |

To go back to the old LM Studio behaviour:

```powershell
$env:API_BASE_URL="http://localhost:1234/v1"
$env:MODEL_NAME="coreocr-7b-050325-preview"
.\venv312\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 11. Troubleshooting

### `422 Unprocessable Entity` with `"loc": ["body","file"]`

The form field has the wrong name. Use `file` or `image`, and make sure the row's type is **File**,
not Text.

### `400 Unsupported type 'application/octet-stream'`

This was fixed — the service now detects the format from the file bytes. If you still see it, you are
running an old copy of `main.py`; restart the server.

### `503 Backend not reachable` / `Engine process is not running`

Read `engine.log` in the project folder. The most common causes are a missing model file, a GPU
already fully occupied by another process, or a corrupted GGUF download.

### Engine exits immediately, `engine.log` mentions a missing DLL

The CUDA runtime DLLs live in a separate `vendor` folder, not next to `llama-server.exe`. `engine.py`
puts both directories on `PATH` for the child process. If you overrode `LLAMA_SERVER` with a binary
from elsewhere, you must ensure its DLLs are reachable yourself.

### `SSL: WRONG_VERSION_NUMBER` during `pip install`

The configured PyPI mirror is down. Add `--index-url https://pypi.org/simple` — see
[section 5](#5-installation).

### `did not find executable at ...\Python314\python.exe`

You are using the old `venv\` folder. It was built with Python 3.14 on a path that no longer exists
(`D:\ocr`), and Python 3.14 has since been uninstalled. Use `venv312\` instead; the old `venv\` can be
deleted.

### Port 8000 already in use

```powershell
.\venv312\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8080
```

### Leftover engine process after a hard kill

If you kill uvicorn with the task manager instead of `Ctrl + C`, the engine can survive:

```powershell
Get-Process llama-server | Stop-Process -Force
```

### `test_ocr.py` fails

That file predates the current architecture. It imports `cv2` and `numpy` (leftovers from the old
EasyOCR version) and asserts response fields that no longer exist. It has not been updated.

---

## 12. Accuracy notes and tuning

### Image tokens matter — a lot

llama.cpp warns at load time:

```
load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly
```

This is not cosmetic. With the default (lower) value, the model started repeating the same sentence
over and over, ran into the 2048-token output limit, and produced truncated JSON — a `422` on every
request. `IMAGE_MIN_TOKENS=1024` is therefore set as the default in `engine.py` and should not be
lowered.

### Output varies between identical runs

Even at `temperature=0.0`, two identical requests can produce slightly different body text — for
example `51096365` versus `۵۱۰۹۶۳۵`, or a differently worded opening sentence. This is a limitation
of the Q4-quantized model reading Persian script, not a bug in the service.

If you need higher fidelity, in rough order of cost/benefit:

1. Raise `IMAGE_MIN_TOKENS` (e.g. `2048`) — slower, uses more context.
2. Use a larger quantization of the same model (Q5/Q6/Q8 instead of Q4_K_S).
3. Feed higher-resolution scans; a sharp 300 DPI scan beats a phone photo.

### GPU memory

The Q4_K_S model plus the f16 vision projector fit comfortably in 12 GB with a 16384-token context.
If you raise `CONTEXT_SIZE` substantially and hit out-of-memory errors, lower it again or reduce
`N_GPU_LAYERS`.

---

## 13. Change history

### v4.0.0 — embedded engine

- **Removed the dependency on LM Studio running.** New `engine.py` locates the llama.cpp binary and
  the GGUF model files and starts the engine itself as a child process, tied to the app's lifespan.
- The engine is reused if one is already listening on the configured port, and left alone on shutdown
  in that case.
- `POST /ocr` now accepts the form field name `image` as well as `file`.
- Image format is detected from the file bytes instead of trusting the client's Content-Type header,
  so `application/octet-stream` uploads from Postman work.
- `IMAGE_MIN_TOKENS=1024` and `CONTEXT_SIZE=16384` set as defaults to stop the model from looping and
  producing truncated JSON.
- `API_BASE_URL` added as an escape hatch to attach to an external OpenAI-compatible server.
- The previous version is preserved as `main_lmstudio_client.py.bak`.

### v3.0.0 — LM Studio client

- FastAPI service calling LM Studio's local server at `http://localhost:1234/v1`.
- Required the LM Studio application to be open with the model loaded and the server started.

### Earlier — EasyOCR

- The original implementation used EasyOCR + OpenCV and returned per-segment text with confidence
  scores and bounding boxes. Replaced by the vision-language model approach, which returns structured
  fields instead of raw segments. `test_ocr.py` still dates from this era.

---

## Environment as tested

| Item | Value |
|---|---|
| Project path | `E:\ghasemi\ocr` |
| Python | 3.12.10 (`venv312`) |
| GPU | NVIDIA GeForce RTX 5070, 12 GB |
| Engine | llama.cpp `llama-server` 2.28.2, CUDA 12 build |
| Model | coreOCR-7B-050325-preview, Q4_K_S + mmproj-f16 |
| Response time | 3–5 s per letter |
