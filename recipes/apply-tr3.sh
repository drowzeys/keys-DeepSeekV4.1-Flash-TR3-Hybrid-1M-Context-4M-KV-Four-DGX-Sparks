#!/usr/bin/env bash
# Recipe C: overlay Keys wo_b onto our original TR3-Hybrid stock.
# Four-Spark: prefer  ABLIT=1 bash oneshot.sh  from repo root.
set -euo pipefail
SRC="${SRC:-$HOME/models/DeepSeek-V4.1-Flash-TR3-Hybrid}"
DST="${DST:-$HOME/models/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated}"
OVERLAY="${OVERLAY:-$HOME/dsv41-wo-b-ablit}"
hf download drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid --local-dir "$SRC"
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed --local-dir "$OVERLAY"
python3 "${APPLY:-$(dirname "$0")/../serve/apply_wo_b_graft.py}" \
  --src "$SRC" --wo-b "$OVERLAY/wo_b_l10_35.safetensors" --dst "$DST"
echo "TR3-Hybrid ablit dest=$DST  serve with TR3=1 MODEL_DIR=$(basename "$DST")  or ABLIT=1 bash oneshot.sh"
