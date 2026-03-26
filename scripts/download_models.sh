#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

vocoder_url="https://issai.nu.edu.kz/wp-content/uploads/2022/03/parallelwavegan_male2_checkpoint.zip"
acoustic_url="https://issai.nu.edu.kz/wp-content/uploads/2022/03/kaztts_male2_tacotron2_train.loss.ave.zip"

echo "Downloading vocoder..."
curl -L -o parallelwavegan_male2_checkpoint.zip "$vocoder_url"
echo "Downloading acoustic model..."
curl -L -o kaztts_male2_tacotron2_train.loss.ave.zip "$acoustic_url"

echo "Unzipping..."
unzip -q -o parallelwavegan_male2_checkpoint.zip
unzip -q -o kaztts_male2_tacotron2_train.loss.ave.zip

echo "Done. Expected files:"
echo "  parallelwavegan_male2_checkpoint/checkpoint-400000steps.pkl"
echo "  exp/tts_train_raw_char/config.yaml"
echo "  exp/tts_train_raw_char/train.loss.ave_5best.pth"

