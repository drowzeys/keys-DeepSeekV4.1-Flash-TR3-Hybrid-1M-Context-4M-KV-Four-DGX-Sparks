#!/usr/bin/env bash
# Fail-fast: Mia 2.9bpw attention is EXL3 mul1 K=5, not FP8 wo_b.
set -euo pipefail
SRC="${1:-}"
echo "MiaAI-Lab DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks / Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw"
echo "Do NOT run apply_wo_b_graft.py (FP8 sidecar) on this pack."
echo "Use INSTALL.md recipe D / apply-mia-exl3.sh:"
echo "  https://github.com/drowzeys/keys-DeepSeek-V4.1-Flash-Abliterated-Mia-2x-Spark-EXL3"
if [[ -n "$SRC" && -f "$SRC/model.safetensors.index.json" ]]; then
  python3 - <<PY
import json,sys
from pathlib import Path
p=Path("$SRC")/"model.safetensors.index.json"
wm=json.loads(p.read_text()).get("weight_map",{})
n=sum(1 for k in wm if k.endswith("attn.wo_b.weight"))
packed=sum(1 for k in wm if "wo_b" in k and ("trellis" in k or "suh" in k or "svh" in k))
print(f"index: fp8 wo_b.weight count={n} packed-wo_b-ish={packed}")
if n==0 and packed:
    print("CONFIRMED: EXL3 packed wo_b — use mia_exl3_wo_b_l10_35.safetensors")
    sys.exit(0)
if n==0:
    print("no FP8 attn.wo_b.weight — still not the FP8 overlay")
    sys.exit(2)
print("unexpected: FP8 wo_b present; do not assume it matches the Mia sidecar.")
sys.exit(1)
PY
fi
exit 0
