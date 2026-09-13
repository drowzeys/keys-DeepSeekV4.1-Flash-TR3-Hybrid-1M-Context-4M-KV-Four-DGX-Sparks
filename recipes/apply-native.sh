#!/usr/bin/env bash
# Recipe A: overlay Keys wo_b onto original native stock.
set -euo pipefail
SRC="${SRC:-$HOME/models/DeepSeek-V4.1-Flash}"
DST="${DST:-$HOME/models/DeepSeek-V4.1-Flash-Abliterated}"
OVERLAY="${OVERLAY:-$HOME/dsv41-wo-b-ablit}"
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir "$SRC"
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed --local-dir "$OVERLAY"
python3 "$OVERLAY/apply_wo_b_graft.py" --src "$SRC" --wo-b "$OVERLAY/wo_b_l10_35.safetensors" --dst "$DST"
echo "native ablit dest=$DST  serve with TR3=0 MODEL_DIR=$(basename "$DST")"
