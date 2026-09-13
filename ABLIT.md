# Keys abliteration (anchored tensors) — V4.1 Flash TR3-Hybrid

Same family as the published 0731 / Vision-EXP packs. **Not** a dealignai graft.
mHC-resistant: direct FP8 `attn.wo_b` projection, recaptured **5120-d** rank-1
direction on this TR3 (0731's 4096-d vector does not transfer).

## Recipe

| | |
|---|---|
| Direction | recapture harmful (QuantTrio 32 + cyber offense/defense) vs harmless on stock TR3 |
| Edit | `layers.10–35.attn.wo_b` only · λ=**3.5** · k=**1** · 26 tensors · mean Δrel ≈ 0.053 |
| **Stock anchors** | L0–9 · **L36–39** (DSpark targets 37–39) · **all MTP** · vision · Engram · TR3 experts · embed/head |
| Why last layers stock | Projecting through DSpark target layers makes the **stock drafter keep proposing refusal tokens**; accept dies. Same lesson as 0731 L10–35 Anchored-Tensors. |
| Block / dtype | FP8 e4m3 + UE8M0 **32×32** (not 0731's 128×128) |

## Gates (this fleet, 1M champion serve)

- refusal32 **32/32** bypass · 0 refuse · 0 garble
- cyber offense+defense **22/22** bypass · 0 refuse · 0 garble
- ping `SPARK-TR3-OK` · 17×19=323
- Serve knobs unchanged: GMU 0.80, 1M, seqs 8, decode `block_m=8`, FAST_MATH, DSpark k=5

## Weights

Gated, automatic approval after you agree:

https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated-Cybersecurity-Unleashed

```bash
hf download drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated-Cybersecurity-Unleashed \
  --local-dir ~/models/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated
MODEL_DIR=DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated bash oneshot.sh
# or: MODEL_DIR=... python3 cluster.py start  with the champion knobs
```

See [HERMES.md](HERMES.md) before the first Telegram turn.
