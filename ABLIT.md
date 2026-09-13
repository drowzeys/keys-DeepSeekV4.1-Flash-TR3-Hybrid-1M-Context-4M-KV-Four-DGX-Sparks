# Keys abliteration overlay (universal)

Not a second 410 GB checkpoint. Hugging Face:

**https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed**

(~1.1 GB `wo_b_l10_35.safetensors` + `apply_wo_b_graft.py`, gated with automatic approval)

## Original stock bases

| Pack | Hugging Face |
|---|---|
| **Native** | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| **EXL3 3.5 bpw Pollard** | [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) |
| **Our TR3-Hybrid** | [`drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid) |

Attention `wo_b` is byte-identical on all three stocks. Overlay L10–35 only; experts stay that pack.

Full copy-paste for **native**, **EXL3 3.5 bpw**, and **our TR3**: **[INSTALL.md](INSTALL.md)**. Wrappers: `recipes/apply-native.sh`, `recipes/apply-exl3.sh`, `recipes/apply-tr3.sh`.

[MiaAI-Lab 2× Spark EXL3 2.9 bpw](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) is **not** compatible (attention is EXL3 K=5). INSTALL.md recipe **D**.

## Four-Spark TR3 (this repo)

```bash
# stock TR3 already fetched by oneshot, or SKIP_WEIGHTS=1 if on disk
ABLIT=1 bash oneshot.sh
```

That downloads the sidecar from HF onto node `.3`, writes
`~/models/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated` (hardlink pack), and serves `MODEL_DIR` of that dest with the champion knobs.

## Native or EXL3 on your own box

```bash
hf download drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed --local-dir ~/dsv41-wo-b-ablit
python3 ~/dsv41-wo-b-ablit/apply_wo_b_graft.py \
  --src ~/models/DeepSeek-V4.1-Flash \
  --wo-b ~/dsv41-wo-b-ablit/wo_b_l10_35.safetensors \
  --dst ~/models/DeepSeek-V4.1-Flash-Abliterated
```

Full copy-paste: the HF card [APPLY_NATIVE_EXL3.md](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed/blob/main/APPLY_NATIVE_EXL3.md).

## Recipe

| | |
|---|---|
| Direction | recapture 5120-d rank-1 on TR3 (QuantTrio 32 + cyber offense/defense) |
| Edit | `layers.10–35.attn.wo_b` · λ=**3.5** · k=**1** · 26 tensors · mean Δrel ≈ 0.053 |
| **Stock anchors** | L0–9 · L36–39 (DSpark 37–39) · all MTP · vision · Engram · **all experts** |
| Gates | refusal32 **32/32** · cyber **22/22** · 0 garble |

See [HERMES.md](HERMES.md) before the first Telegram turn.
