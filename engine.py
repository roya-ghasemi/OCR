"""
Embedded llama.cpp engine manager.

The OCR service does not need LM Studio to be running — it only needs the GGUF
model files and a llama.cpp `llama-server` binary. Both are already on disk
(LM Studio ships the binaries under ~/.lmstudio/extensions/backends), so this
module locates them, starts the engine as a child process, and shuts it down
again when the app stops.
"""
import logging
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Single source of truth for every path and knob. This module used to
# re-read the same environment variables itself, so config.py and engine.py
# could disagree about where the model lives — and did, the moment the GGUFs
# moved off C:.
import config

logger = logging.getLogger(__name__)

LMSTUDIO_ROOT = config.LMSTUDIO_ROOT
BACKENDS_DIR = LMSTUDIO_ROOT / "extensions" / "backends"   # backends stayed on C:
MODELS_DIR = config.MODELS_DIR                             # models moved to D:

# Backend flavours in descending order of preference; each maps to the vendor
# folder holding its runtime DLLs (CUDA libs are not next to the executable).
_BACKEND_PREFERENCE = [
    ("llama.cpp-win-x86_64-nvidia-cuda12-avx2-", "win-llama-cuda12-vendor-v2"),
    ("llama.cpp-win-x86_64-nvidia-cuda-avx2-", "win-llama-cuda-vendor-v2"),
    ("llama.cpp-win-x86_64-vulkan-avx2-", "win-llama-vulkan-vendor-v2"),
    ("llama.cpp-win-x86_64-avx2-", None),
]

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)$")


def _version_key(name: str) -> tuple:
    m = _VERSION_RE.search(name)
    return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)


def find_backend() -> tuple[Path, Optional[Path]]:
    """Return (llama-server.exe, vendor_dll_dir) for the best available backend."""
    override = config.LLAMA_SERVER
    if override:
        exe = Path(override)
        if not exe.is_file():
            raise FileNotFoundError(f"LLAMA_SERVER does not point at a file: {exe}")
        return exe, None

    if not BACKENDS_DIR.is_dir():
        raise FileNotFoundError(
            f"No llama.cpp backends found at {BACKENDS_DIR}. "
            f"Set LLAMA_SERVER to a llama-server executable."
        )

    pin = (config.ENGINE_BUILD_PIN or "").strip()

    for prefix, vendor_name in _BACKEND_PREFERENCE:
        candidates = sorted(
            (d for d in BACKENDS_DIR.iterdir()
             if d.is_dir() and d.name.startswith(prefix) and (d / "llama-server.exe").is_file()),
            key=lambda d: _version_key(d.name),
            reverse=True,
        )
        if not candidates:
            continue

        # Prefer the pinned build. Taking the newest would silently change the
        # engine under a fixed ENGINE_BUILD provenance string.
        chosen = None
        if pin:
            chosen = next((d for d in candidates if d.name.endswith(f"-{pin}")), None)
            if chosen is None:
                logger.warning(
                    "Pinned backend %s not found under %s; falling back to %s. "
                    "Results from this run are NOT comparable to earlier ones — "
                    "update config.ENGINE_BUILD and re-baseline, or install the "
                    "pinned build.", pin, BACKENDS_DIR, candidates[0].name,
                )
        if chosen is None:
            chosen = candidates[0]

        vendor = BACKENDS_DIR / "vendor" / vendor_name if vendor_name else None
        if vendor is not None and not vendor.is_dir():
            vendor = None
        return chosen / "llama-server.exe", vendor

    raise FileNotFoundError(
        f"No usable llama-server.exe under {BACKENDS_DIR}. "
        f"Set LLAMA_SERVER to a llama-server executable."
    )


def find_model() -> tuple[Path, Optional[Path]]:
    """Return (model_gguf, mmproj_gguf) — explicit config wins, else search MODELS_DIR."""
    explicit = config.MODEL_GGUF
    if explicit:
        model = Path(explicit)
        if not model.is_file():
            raise FileNotFoundError(f"MODEL_GGUF does not point at a file: {model}")
    else:
        pattern = config.MODEL_GLOB
        matches = [p for p in MODELS_DIR.glob(pattern) if "mmproj" not in p.name.lower()]
        if not matches:
            # Name every root that was searched. The previous message named only
            # the resolved directory, which after a move is the one place the
            # file is guaranteed not to be.
            searched = "\n  ".join(str(r) for r in config.MODEL_SEARCH_ROOTS)
            raise FileNotFoundError(
                f"No model matching '{pattern}' under {MODELS_DIR}.\n"
                f"Roots searched:\n  {searched}\n"
                f"Set MODEL_GGUF to the .gguf file, or MODELS_DIR / "
                f"MODEL_SEARCH_ROOTS to the directory holding it."
            )
        model = max(matches, key=lambda p: p.stat().st_size)

    mmproj_env = config.MMPROJ_GGUF
    if mmproj_env:
        mmproj = Path(mmproj_env)
        if not mmproj.is_file():
            raise FileNotFoundError(f"MMPROJ_GGUF does not point at a file: {mmproj}")
    else:
        siblings = [p for p in model.parent.glob("*.gguf") if "mmproj" in p.name.lower()]
        mmproj = siblings[0] if siblings else None

    return model, mmproj


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _models_endpoint_ready(base_url: str) -> bool:
    """llama-server answers /v1/models with 503 while the weights are still loading."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            return resp.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


class LlamaEngine:
    """Owns the llama-server child process for the lifetime of the app."""

    def __init__(self) -> None:
        self.host = config.ENGINE_HOST
        self.port = config.ENGINE_PORT
        self.process: Optional[subprocess.Popen] = None
        self.log_handle = None
        self.model_path: Optional[Path] = None
        self.mmproj_path: Optional[Path] = None
        self.adopted = False          # engine was already running; we did not start it
        self.log_path = Path(config.ENGINE_LOG).resolve()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        if _port_is_open(self.host, self.port):
            logger.info("Engine already listening on %s:%s — reusing it.", self.host, self.port)
            self.adopted = True
            self._wait_until_ready()
            return

        exe, vendor = find_backend()
        self.model_path, self.mmproj_path = find_model()

        cmd = [
            str(exe),
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", config.N_GPU_LAYERS,
            "-c", config.CONTEXT_SIZE,
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", str(self.mmproj_path)]
            # Qwen-VL needs at least 1024 image tokens; below that the model
            # starts repeating itself and the JSON never terminates.
            cmd += ["--image-min-tokens", config.IMAGE_MIN_TOKENS]

        # The CUDA/Vulkan runtime DLLs live in a sibling vendor folder, and the
        # backend's own DLLs sit next to the executable — both must be on PATH.
        env = os.environ.copy()
        extra_paths = [str(exe.parent)] + ([str(vendor)] if vendor else [])
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])

        logger.info("Starting engine: %s", exe)
        logger.info("Model:  %s", self.model_path)
        logger.info("mmproj: %s", self.mmproj_path or "(none - text only)")
        logger.info("Engine log: %s", self.log_path)

        self.log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(exe.parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        timeout = config.ENGINE_STARTUP_TIMEOUT
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Engine exited with code {self.process.returncode}. "
                    f"See {self.log_path} for details."
                )
            if _models_endpoint_ready(self.base_url):
                logger.info("Engine ready at %s", self.base_url)
                return
            time.sleep(2)

        self.stop()
        raise RuntimeError(f"Engine did not become ready within {timeout:.0f}s. See {self.log_path}.")

    def stop(self) -> None:
        if self.process is None or self.adopted:
            return
        logger.info("Stopping engine (pid %s).", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("Engine did not stop in time - killing it.")
            self.process.kill()
        finally:
            self.process = None
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None

    def is_alive(self) -> bool:
        if self.adopted:
            return _port_is_open(self.host, self.port)
        return self.process is not None and self.process.poll() is None
