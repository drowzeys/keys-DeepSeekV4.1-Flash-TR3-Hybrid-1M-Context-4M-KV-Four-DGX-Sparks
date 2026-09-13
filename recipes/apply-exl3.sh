#!/usr/bin/env bash
# Recipe B: overlay Keys wo_b onto original EXL3 3.5 bpw Pollard stock.
set -euo pipefail
SRC="${SRC:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-Pollard}"
DST="${DST:-$HOME/models/DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated}"
OVERLAY="${OVERLAY:-$HOME/dsv41-wo-b-ablit}"
hf download bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard --local-dir "$SRC"
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed --local-dir "$OVERLAY"
python3 "$OVERLAY/apply_wo_b_graft.py" --src "$SRC" --wo-b "$OVERLAY/wo_b_l10_35.safetensors" --dst "$DST"
echo "EXL3 3.5bpw ablit dest=$DST  serve with TR3=0 MODEL_DIR=$(basename "$DST") (not TR3=1)"
