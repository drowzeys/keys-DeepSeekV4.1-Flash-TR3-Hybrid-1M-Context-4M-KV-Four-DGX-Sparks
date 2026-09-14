#!/usr/bin/env bash
# Recipe D: overlay Keys EXL3-K5 wo_b onto Mia 2.9bpw 2x Spark stock.
set -euo pipefail
SRC="${SRC:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
DST="${DST:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw-Abliterated}"
OVERLAY="${OVERLAY:-$HOME/dsv41-wo-b-ablit}"
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw --local-dir "$SRC"
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed \
  --include "mia_exl3_wo_b_l10_35.safetensors" --include "apply_mia_exl3_wob.py" \
  --local-dir "$OVERLAY"
python3 "$OVERLAY/apply_mia_exl3_wob.py" \
  --src "$SRC" --wo-b "$OVERLAY/mia_exl3_wo_b_l10_35.safetensors" --dst "$DST"
echo "Mia 2.9bpw ablit dest=$DST"
echo "Set Mia kit MODEL_HOST=$DST (keep ENGRAM_DIR as native 47+48). Do not use apply_wo_b_graft.py."
