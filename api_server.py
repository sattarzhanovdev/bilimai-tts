#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent



def _resolve_path(p: str) -> str:
    path = Path(p).expanduser()
    if path.is_absolute():
        return str(path)
    return str((BASE_DIR / path).resolve())


def _ensure_parallel_wavegan_scipy_compat() -> None:
    # parallel-wavegan<=0.6.1 expects scipy.signal.kaiser, but in newer SciPy it's under scipy.signal.windows.
    try:
        import scipy.signal as scipy_signal

        if hasattr(scipy_signal, "kaiser"):
            return

        from scipy.signal.windows import kaiser as kaiser_win  # type: ignore

        setattr(scipy_signal, "kaiser", kaiser_win)
    except Exception:
        # If SciPy is missing/broken, the vocoder import will fail later with a clear error.
        return


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    # Optional, but keeps the API flexible.
    lang: str | None = Field(default=None, description="kyrgyz, kazakh, turkish, ...")


class TTSResponse(BaseModel):
    id: str
    audio_url: str


class TurkicTTSService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._init_error: str | None = None

        self.device = os.getenv("TURKICTTS_DEVICE", "cpu")
        self.default_lang = os.getenv("TURKICTTS_LANG", "kyrgyz")
        self.fs = int(os.getenv("TURKICTTS_FS", "22050"))

        self.vocoder_checkpoint = os.getenv(
            "TURKICTTS_VOCODER",
            "parallelwavegan_male2_checkpoint/checkpoint-400000steps.pkl",
        )
        self.config_file = os.getenv("TURKICTTS_CONFIG", "exp/tts_train_raw_char/config.yaml")
        self.model_path = os.getenv("TURKICTTS_MODEL", "exp/tts_train_raw_char/train.loss.ave_5best.pth")

        self._vocoder = None
        self._text2speech = None

    def init(self) -> None:
        if self._ready or self._init_error is not None:
            return

        try:
            _ensure_parallel_wavegan_scipy_compat()

            from parallel_wavegan.utils import load_model
            from espnet2.bin.tts_inference import Text2Speech
            import torch

            from utils import normalization

            self._normalization = normalization  # type: ignore[attr-defined]

            self.vocoder_checkpoint = _resolve_path(self.vocoder_checkpoint)
            self.config_file = _resolve_path(self.config_file)
            self.model_path = _resolve_path(self.model_path)

            missing = [p for p in [self.vocoder_checkpoint, self.config_file, self.model_path] if not Path(p).exists()]
            if missing:
                raise RuntimeError(
                    "Не найдены файлы моделей. Ожидаются:\n"
                    f"  - {self.vocoder_checkpoint}\n"
                    f"  - {self.config_file}\n"
                    f"  - {self.model_path}\n"
                    "Скачайте и распакуйте pretrained модели из README.md в корень проекта."
                )

            device = self.device
            if device != "cpu" and not torch.cuda.is_available():
                device = "cpu"

            vocoder = load_model(self.vocoder_checkpoint).to(device).eval()
            vocoder.remove_weight_norm()

            t2s = Text2Speech(
                self.config_file,
                self.model_path,
                device=device,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=10.0,
                use_att_constraint=True,
                backward_window=1,
                forward_window=3,
                speed_control_alpha=1.0,
            )
            t2s.spc2wav = None

            self._torch = torch
            self._load_model = load_model
            self._Text2Speech = Text2Speech
            self._vocoder = vocoder
            self._text2speech = t2s
            self._device = device
            self._ready = True
        except Exception as e:
            self._init_error = str(e)

    def synth(self, text: str, lang: str | None) -> "tuple[int, object]":
        self.init()
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        assert self._vocoder is not None and self._text2speech is not None

        lang = (lang or self.default_lang).strip().lower()
        normalized = self._normalization(text, lang)  # type: ignore[misc]

        with self._lock:
            with self._torch.no_grad():
                c_mel = self._text2speech(normalized)["feat_gen"]
                wav = self._vocoder.inference(c_mel)

        return self.fs, wav.view(-1).cpu().numpy()


app = FastAPI(title="TurkicTTS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_dir_env = os.getenv("TURKICTTS_AUDIO_DIR", "audio")
audio_dir = Path(audio_dir_env).expanduser()
if not audio_dir.is_absolute():
    audio_dir = BASE_DIR / audio_dir
audio_dir = audio_dir.resolve()
audio_dir.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")

tts = TurkicTTSService()

@app.get("/")
def root() -> dict:
    return {"status": "ok"}

@app.get("/health")
def health() -> dict:
    return {"ok": True}


# @app.post("/tts", response_model=TTSResponse)
# def tts_endpoint(payload: TTSRequest, request: Request) -> TTSResponse:
#     text = payload.text.strip()
#     if not text:
#         raise HTTPException(status_code=400, detail="text is empty")

#     try:
#         fs, wav = tts.synth(text=text, lang=payload.lang)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     file_id = f"{int(time.time())}_{uuid.uuid4().hex[:12]}"
#     filename = f"{file_id}.wav"
#     out_path = audio_dir / filename

#     from scipy.io.wavfile import write as wav_write

#     wav_write(str(out_path), fs, wav)

#     base = str(request.base_url).rstrip("/")
#     return TTSResponse(id=file_id, audio_url=f"{base}/audio/{filename}")

@app.post("/tts")
def tts_endpoint(payload: TTSRequest, request: Request):
    return {"text": payload.text, "status": "received"}