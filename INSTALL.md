# Installation recipes — apply the ablit the right way for your pack

One overlay: [`drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed)  
(`wo_b_l10_35.safetensors` ~1.1 GB, gated, automatic approval)

Pick **your original stock**. Do **not** mix experts (do not graft TR3 experts onto native, etc.). Only `layers.10–35.attn.wo_b` changes. Never write stock in place.

| Recipe | Original stock | Dest after overlay | Serve |
|---|---|---|---|
| **A — Native** | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `~/models/DeepSeek-V4.1-Flash-Abliterated` | native MXFP4 MoE, `TR3=0` |
| **B — EXL3 3.5 bpw** | [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | `~/models/DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated` | Pollard EXL3 MoE, `TR3=0` |
| **C — Our TR3-Hybrid** | [`drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid) | `~/models/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated` | TR3 plugin, `ABLIT=1 bash oneshot.sh` |
| **D — Mia 2× Spark EXL3 2.9 bpw** | [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) + [2× Spark kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | `model-ablit/` next to their `model/` | **EXL3 mul1 K=5 sidecar** `mia_exl3_wo_b_l10_35.safetensors` (not the FP8 overlay) |

GPU memory utilization **≤ 0.85**. Shared overlay download:

```bash
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed \
  --local-dir ~/dsv41-wo-b-ablit
pip install torch safetensors   # or use the vLLM image python
```

---

## A — Native (original stock)

MXFP4 routed experts stay native. Official FP8 attention except L10–35 `wo_b`.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir ~/models/DeepSeek-V4.1-Flash

python3 ~/dsv41-wo-b-ablit/apply_wo_b_graft.py \
  --src ~/models/DeepSeek-V4.1-Flash \
  --wo-b ~/dsv41-wo-b-ablit/wo_b_l10_35.safetensors \
  --dst ~/models/DeepSeek-V4.1-Flash-Abliterated
```

**Serve (this 4× Spark launcher):** same image as stock native, **no** TR3 plugin.

```bash
cd serve
IMAGE=vllm-dsv41:overlay5-e47aa PATCH_SET=patch-upstream-boot10 \
  TR3=0 MODEL_DIR=DeepSeek-V4.1-Flash-Abliterated \
  GMU=0.81 MAXLEN=1048576 SEQS=8 BATCH=2048 EAGER=0 SPEC=1 \
  python3 cluster.py start
```

If you already have a native recipe, keep every flag and only change the checkpoint path / `MODEL_DIR` to the **Abliterated** dest.

---

## B — EXL3 3.5 bpw Pollard (original stock)

Pollard EXL3 experts stay Pollard. Do **not** use `TR3=1` (that plugin expects the TR3-Hybrid keep-64 layout).

```bash
hf download bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard \
  --local-dir ~/models/DeepSeek-V4.1-Flash-EXL3-Pollard

python3 ~/dsv41-wo-b-ablit/apply_wo_b_graft.py \
  --src ~/models/DeepSeek-V4.1-Flash-EXL3-Pollard \
  --wo-b ~/dsv41-wo-b-ablit/wo_b_l10_35.safetensors \
  --dst ~/models/DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated
```

**Serve:** point your existing Pollard EXL3 stack at the dest. On this launcher:

```bash
cd serve
IMAGE=vllm-dsv41:overlay5-e47aa PATCH_SET=patch-upstream-boot10 \
  TR3=0 MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated \
  GMU=0.81 MAXLEN=1048576 SEQS=8 BATCH=2048 EAGER=0 SPEC=1 \
  python3 cluster.py start
```

`CUDA_EXL3_MODEL_PATH` is set from `MODEL_DIR`. Keep any extra EXL3 patches you already used for stock Pollard.

---

## C — Our TR3-Hybrid (original stock)

K3 tail + 64 MXFP4 keeps stay TR3. This is the pack `oneshot.sh` fetches.

```bash
git clone https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option
cd keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option

# stock TR3 + overlay + champion 1M serve (decode block_m=8, FAST_MATH, DSpark k=5)
ABLIT=1 bash oneshot.sh

# stock already on disk:
SKIP_WEIGHTS=1 ABLIT=1 bash oneshot.sh
```

Manual overlay (same as oneshot `ABLIT=1`):

```bash
hf download drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid --local-dir ~/models/DeepSeek-V4.1-Flash-TR3-Hybrid
python3 serve/apply_wo_b_graft.py \
  --src ~/models/DeepSeek-V4.1-Flash-TR3-Hybrid \
  --wo-b ~/dsv41-wo-b-ablit/wo_b_l10_35.safetensors \
  --dst ~/models/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated
```

**Serve:** `TR3=1 MODEL_DIR=DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated` (oneshot does this).

---

## Check

After overlay, `dst/ABLIT_META.json` should show `"n_edited": 52` (26 layers × weight+scale) or `"n_edited": 26` depending on count of `.weight` only in older prints — current script counts both. L0–9 / L36–39 / MTP / experts must remain hardlinked or byte-identical to `--src`.

Hermes: [HERMES.md](HERMES.md) (`max_tokens` 12288, `reasoning_effort: false`, warmup before first Telegram turn).

---

## D — MiaAI-Lab 2× DGX Spark EXL3 2.9 bpw

Kit: [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)  
Stock weights: [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)

Attention is **EXL3 mul1 K=5**, not FP8. Do **not** use `apply_wo_b_graft.py`. Use the **Mia sidecar** (dequant → Keys λ=3.5 project → requant mul1 K=5; roundtrip NMSE ≈ 0.0012).

```bash
# 1) stock Mia pack (or reuse ./model from their start.sh)
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw --local-dir ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw

# 2) overlay only (~650 MB)
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed \
  --include "mia_exl3_wo_b_l10_35.safetensors" \
  --include "apply_mia_exl3_wob.py" \
  --local-dir ~/dsv41-wo-b-ablit

python3 ~/dsv41-wo-b-ablit/apply_mia_exl3_wob.py \
  --src ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --wo-b ~/dsv41-wo-b-ablit/mia_exl3_wo_b_l10_35.safetensors \
  --dst ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw-Abliterated
```

**Serve with their kit** — keep `start.sh` / image / Engram-from-native. Point `MODEL_HOST` at the **Abliterated** dest (not stock `./model`):

```bash
cd DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
# .env
# MODEL_HOST=/home/you/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw-Abliterated
# ENGRAM_DIR still the native shards 47+48 as in their README
./start.sh
```

Wrapper: `recipes/apply-mia-exl3.sh`. Engram stays native (untouched). L0–9 / L36–39 / MTP `wo_b` stay stock EXL3.
