#!/usr/bin/env bash
# Fail-fast: Mia 2.9bpw 2x Spark EXL3 is NOT compatible with the FP8 wo_b overlay.
set -euo pipefail
SRC="${1:-}"
echo "MiaAI-Lab DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks / Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw"
echo "This Keys overlay is FP8 layers.10-35 attn.wo_b."
echo "That pack quantizes attention to EXL3 K=5 mul1 (exl3_k_map.json attn_default=5)."
echo "apply_wo_b_graft.py MUST NOT be run on it."
echo "Use INSTALL.md recipe B (Pollard 3.5bpw) or A (native) instead."
if [[ -n "$SRC" && -f "$SRC/model.safetensors.index.json" ]]; then
  python3 - <<PY
import json,sys
from pathlib import Path
p=Path("$SRC")/"model.safetensors.index.json"
wm=json.loads(p.read_text()).get("weight_map",{})
n=sum(1 for k in wm if k.endswith("attn.wo_b.weight"))
packed=sum(1 for k in wm if "wo_b" in k and ("trellis" in k or "suh" in k or "svh" in k))
print(f"index: fp8 wo_b.weight count={n} packed-wo_b-ish={packed}")
if n==0:
    print("CONFIRMED: no FP8 attn.wo_b.weight — overlay incompatible.")
    sys.exit(2)
print("unexpected: FP8 wo_b present; still do not assume K=5 mul1 matches our sidecar.")
sys.exit(1)
PY
fi
exit 2
