#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _missing_deps_message() -> str:
    return (
        "Не найдены зависимости для TTS (espnet2 и/или parallel_wavegan).\n"
        "Установите их в активированное окружение:\n"
        "  python -m pip install -U pip\n"
        "  python -m pip install espnet parallel-wavegan\n"
        "\n"
        "Примечание: для CPU достаточно torch (у вас он уже есть), но установка espnet может занять время."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="TurkicTTS inference helper (ESPnet2 + ParallelWaveGAN).")
    parser.add_argument("--text", required=True, help="Input text (e.g. merhaba)")
    parser.add_argument(
        "--lang",
        default="turkish",
        help="One of: azerbaijani,bashkir,kazakh,kyrgyz,sakha,tatar,turkish,turkmen,uyghur,uzbek",
    )
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--out", default="result.wav", help="Output wav path")
    parser.add_argument(
        "--fs",
        type=int,
        default=22050,
        help="Sample rate for output wav (matches README example)",
    )
    parser.add_argument(
        "--vocoder-checkpoint",
        default="parallelwavegan_male2_checkpoint/checkpoint-400000steps.pkl",
        help="Path to ParallelWaveGAN checkpoint .pkl",
    )
    parser.add_argument(
        "--config",
        default="exp/tts_train_raw_char/config.yaml",
        help="Path to ESPnet2 TTS config.yaml",
    )
    parser.add_argument(
        "--model",
        default="exp/tts_train_raw_char/train.loss.ave_5best.pth",
        help="Path to ESPnet2 TTS model .pth",
    )
    args = parser.parse_args()

    try:
        from parallel_wavegan.utils import load_model
        from espnet2.bin.tts_inference import Text2Speech
        import torch
        import scipy.signal as _scipy_signal
        from scipy.io.wavfile import write as wav_write
    except Exception:
        print(_missing_deps_message(), file=sys.stderr)
        return 2

    # parallel-wavegan<=0.6.1 expects scipy.signal.kaiser, but in newer SciPy it's under scipy.signal.windows.
    if not hasattr(_scipy_signal, "kaiser"):
        try:
            from scipy.signal.windows import kaiser as _kaiser  # type: ignore

            setattr(_scipy_signal, "kaiser", _kaiser)
        except Exception:
            pass

    try:
        from utils import normalization
    except Exception as e:
        print(f"Не удалось импортировать normalization из utils.py: {e}", file=sys.stderr)
        return 2

    vocoder_checkpoint = Path(args.vocoder_checkpoint)
    config_file = Path(args.config)
    model_path = Path(args.model)

    missing = [p for p in [vocoder_checkpoint, config_file, model_path] if not p.exists()]
    if missing:
        print("Не найдены файлы модели. Ожидаются (по умолчанию):", file=sys.stderr)
        for p in [vocoder_checkpoint, config_file, model_path]:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nСкачайте и распакуйте pretrained модели из README.md в корень проекта так, чтобы появились папки "
            "`parallelwavegan_male2_checkpoint/` и `exp/tts_train_raw_char/`.",
            file=sys.stderr,
        )
        return 2

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA не доступна, переключаюсь на cpu.", file=sys.stderr)
        device = "cpu"

    vocoder = load_model(str(vocoder_checkpoint)).to(device).eval()
    vocoder.remove_weight_norm()

    text2speech = Text2Speech(
        str(config_file),
        str(model_path),
        device=device,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None  # disable griffin-lim

    text = normalization(args.text, args.lang)
    with torch.no_grad():
        c_mel = text2speech(text)["feat_gen"]
        wav = vocoder.inference(c_mel)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_write(str(out_path), args.fs, wav.view(-1).cpu().numpy())
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
