# DeepSeek-V4.1-Flash TR3-Hybrid — 1M Context, 8M-token KV pool, Four DGX Sparks, with abliterated option

Serving **[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** (763B, MoE, sparse-MLA + Engram) on **four NVIDIA DGX Spark (GB10, 128 GB unified)** over a 200G ConnectX-7 fabric, tensor-parallel 4, with **CUDA graphs + DSpark speculative decoding + native vision + tool calling + 1M-token context**.

The serving checkpoint is the **TR3-Hybrid** quant: most routed experts are **EXL3-TR3 K=3 (mcg codebook, ~3.0 bpw)**, the 64 highest round-trip-error experts per layer stay **native MXFP4**, everything else (attention, indexer, Engram tables, hyper-connections) is the release's own format. ~3.22 bpw MoE mix.

**Tiers.** *Tier 1* is the EXL3-TR3 **K3 trellis tail** (320 experts/layer, ~3.0 bpw) — the lossy tier that dominates the checkpoint and whose fidelity the KLD measures. *Tier 2* is the **64 keep experts/layer** left as native MXFP4 (bit-identical to native, the quality anchor). Troubleshooting low tier-1 fidelity = raise its precision (K3→K4, or move more high-round-trip-error experts into the native keep tier) and verify with a **large-sample KLD (≥50K teacher-forced positions)**, not a small smoke test.

> **Weights:** https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid

## Quick start (one-shot, four DGX Sparks)

Everything needed to bring the **current-serve** config up in one load:

```bash
git clone https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option
cd keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-8M-KV-Four-DGX-Sparks-with-abliterated-option
bash oneshot.sh            # stock TR3-Hybrid
#   SKIP_WEIGHTS=1 bash oneshot.sh
#   ABLIT=1 bash oneshot.sh          # overlay Keys L10-35 wo_b from HF, then serve that
```

- **Engine image (prebuilt):** `ghcr.io/drowzeys/vllm-dsv41-overlay5-e47aa:serving-node1` (vLLM `dsv41-feat`@e47aa780b + DSpark + TR3/B12X plugin, GB10/sm121). Serves the native and TR3-Hybrid checkpoints.
  - Tag `serving-node1` is **byte-identical to the image serving the champion on node .1** (digest `ad5cc20c`); `:latest` is an equivalent head-node build of the same tree.
  - The package is **private** on first push; `oneshot.sh` authenticates with `gh auth token` (or set `GHCR_TOKEN`). To allow anonymous pulls, flip it to public once at `github.com/users/drowzeys/packages/container/vllm-dsv41-overlay5-e47aa/settings`.
- **Weights:** https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid (~410 GB).
- **Plugin + patches + launcher:** in [`serve/`](serve/) (bind-mounted by `serve/serve-rank.sh`; `cluster.py` fans out to the four ranks).
- Edit the node IPs / model path at the top of `oneshot.sh` for a different cluster.

Current-serve knobs: `TP=4, GMU 0.80, max-model-len 1048576, max-num-seqs 8, CUDA graphs FULL_AND_PIECEWISE, DSpark k=5, fast_math on, decode block_m 8, prefill block_m 64`.

**Hermes:** do not type in Telegram until you have run the warmup — [HERMES.md](HERMES.md). `oneshot.sh` does this after SERVING.

**Abliteration (optional):** three install recipes in **[INSTALL.md](INSTALL.md)** — native / EXL3 3.5 bpw / our TR3. Four-Spark TR3: `ABLIT=1 bash oneshot.sh`. Overlay: [`drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed) (gated, automatic approval).

**Original stock bases**

| Pack | Hugging Face |
|---|---|
| Native | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| EXL3 3.5 bpw Pollard | [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) |
| Our TR3-Hybrid | [`drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid) |
| Mia 2× Spark EXL3 2.9 bpw | [kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) / [`Mia-AiLab/…-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) — recipe **D**, EXL3 K=5 sidecar (not FP8) |

## Headline

- **1M-token context**, KV pool **9,452,923 tokens (8M+ target)** => **~9 concurrent full-1M requests (C=8–9)** on four Sparks.
- Model weights **69.2 GiB/rank** (vs 81.6 GiB/rank for the native release pack), which is what frees the KV room.
- **KV-pool lever:** the pool grew from ~4.5M to **9.45M** by raising `--gpu-memory-utilization` 0.80→0.83 and dropping `--max-num-batched-tokens` 8192→2048 — a smaller prefill chunk shrinks vLLM's activation reserve and hands that memory to the pool. 2048 is the floor with vision on (the multimodal item is 1025 tokens). fp8_ds_mla is the KV-dtype floor for V4.1, so ~12M is the practical pool ceiling at 1M context (and only text-only).
- **1M needle-in-haystack: PASS** (retrieval at depth 0.5 in a >1M-token prompt).
- Quality: tied on an objective battery and **KL 0.032 nats vs native over 66.6K teacher-forced positions (~44% below EXL3's 0.057)** — the tightest-tracking quant here.

## Results

All speed numbers are greedy, temperature 0, measured on this cluster. Native / EXL3 / TR3 rows are matched at 300K context, GMU 0.80; the fourth column is the **deployed** TR3 config (fast_math on, decode block_m 8) measured at 1M context.

### Single-stream decode tok/s (C=1)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — earlier 4M-pool) |
|---|---|---|---|---|
| prose | 27.4 | 23.8 | 30.7 | 26.0 |
| list | 41.8 | 36.0 | 43.7 | 42.0 |
| code | 49.0 | 51.9 | 58.2 | 46.6 |
| essay | 28.9 | 25.3 | 34.4 | 29.9 |
| read | 37.3 | 37.6 | 41.6 | 35.2 |
| math | 56.7 | 49.3 | 64.0 | 48.3 |

### Aggregate throughput tok/s (C=4, wall-clock incl. TTFT)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — earlier 4M-pool) |
|---|---|---|---|---|
| prose | 53.6 | 49.9 | 68.3 | 56.7 |
| list | 70.6 | 91.3 | 111.0 | 101.7 |
| code | 84.1 | 95.9 | 110.1 | 99.1 |
| essay | 57.2 | 60.9 | 78.1 | 68.2 |
| read | 63.8 | 77.4 | 74.5 | 80.5 |
| math | 84.4 | 93.6 | 137.2 | 119.2 |

### Aggregate throughput tok/s (C=8)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — earlier 4M-pool) |
|---|---|---|---|---|
| prose | 73.3 | 79.8 | 88.1 | 75.8 |
| list | 117.0 | 127.8 | 149.4 | 140.9 |
| code | 120.2 | 136.1 | 157.2 | 132.6 |
| essay | 77.7 | 87.5 | 102.2 | 85.2 |
| read | 109.9 | 132.1 | 123.5 | 138.9 |
| math | 153.4 | 173.4 | 189.2 | 165.4 |

### Current-serve C1→C8 sweep (8M-pool config: GMU 0.83, batch 2048, DSpark k=5, vision on)

Greedy-ish (temp 0.7), short prompts, 256 max tokens, measured on the live 9.45M-pool serve. Per-stream is one request's own decode rate; aggregate is the total across all concurrent streams.

**Per-stream decode tok/s**

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| math | 44.6 | 42.6 | 29.3 | 23.3 |
| code | 37.3 | 32.2 | 25.7 | 17.9 |
| read | 35.9 | 32.2 | 27.2 | 16.8 |
| list | 34.5 | 32.1 | 23.5 | 18.6 |
| essay | 23.0 | 20.8 | 15.1 | 10.9 |
| prose | 22.2 | 18.4 | 13.0 | 10.1 |

**Aggregate tok/s (wall, TTFT included)**

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| math | 41.4 | 78.2 | 108.2 | 171.3 |
| list | 33.2 | 59.9 | 89.5 | 135.8 |
| code | 35.7 | 60.3 | 94.7 | 130.2 |
| essay | 22.4 | 40.2 | 58.3 | 81.9 |
| prose | 21.3 | 35.6 | 48.4 | 77.8 |
| read | 26.2 | 46.8 | 74.7 | 74.0 |

TTFT stayed 0.29–0.59 s across the sweep (read at C8 the only outlier, 1.8 s). Aggregate scales ~3.6–4.1× from C1 to C8.

### Memory, KV pool, context

| config | weights GiB/rank | KV pool (tokens) | measured at | max context |
|---|---|---|---|---|
| Native MXFP4 | 81.58 | 1,428,283 | 300K | 1M |
| EXL3 3.5bpw | 56.61 | 3,534,988 | 300K | 1M |
| TR3-Hybrid | 69.21 | 4,190,217 | 300K | 1M |
| TR3-Hybrid (bm8, earlier 4M-pool) | 69.2 | 4,492,902 | 1M | 1M |
| **TR3-Hybrid (8M-pool — CURRENT SERVE)** | 69.2 | **9,452,923** | **1M** | **1M** |

The current-serve pool of **9,452,923 tokens** (9.02× a full 1M request) is measured at **GMU 0.83, `--max-num-batched-tokens` 2048**, vision + tools + graphs + DSpark k=5 all on. That is more than double the earlier 4.5M pool at the same 1M context.

### Intelligence / quality (objective battery + 66.6K-position KLD; native = reference)

| metric | Native | TR3-Hybrid (old) | EXL3 3.5bpw |
|---|---|---|---|
| objective correct | 24/27 | 25/27 | 25/27 |
| KL vs native (nats), 66.6K teacher-forced positions ↓ | 0 | **0.032** | 0.057 |
| top-1 agreement, teacher-forced (66.6K pos) | 1.000 | **0.984** | 0.976 |

> **Reference model.** DeepSeek shipped V4.1-Flash as an **MXFP4-experts / MXFP8** checkpoint (`expert_dtype fp4`, `weight_block_size [32,32]`, `ue8m0` scales); there is no public BF16 original. That shipped checkpoint is what we call **native** and is the **source both quants were made from** — so it is the ground-truth reference here, not a lossless BF16 model. TR3's 64 keep-experts/layer are **bit-identical** to native's MXFP4 experts; its tail is K3-trellis quantized from native. The KL and top-1 figures below measure how faithfully each quant reproduces native's per-token distribution.

The **KL divergence** is the rigorous precision metric, measured to brandonmusic's spec: native's (the teacher's) logits over a representative **32-window / 2047-token corpus (66,599 teacher-forced positions, >50K)**, top-20 truncation, KL(native ‖ quant) averaged per position — no alignment artifact. **TR3's 0.032 nats is ~44% below EXL3's 0.057**, and both track native tightly (top-1 >97.5%). (An earlier 274-position smoke test read 0.259/0.343; that sample was far too small and its numbers swung — hence the large-sample re-run. For scale, a third-party GLM-5.3 EXL3 KLD was ~0.102 nats.) All three are tied on task correctness (objective battery). On the 66.6K-position KLD, TR3-Hybrid reproduces the shipped checkpoint **closer** than EXL3 3.5 bpw — **KL 0.032 vs 0.057, top-1 0.984 vs 0.976** — partly because 64 experts/layer are bit-identical to native.

## Why each config

- **TR3-Hybrid (deployed):** best precision-for-memory on this fabric. Keeps the 64 hardest experts/layer native so quality tracks the full model, while the K3 trellis tail shrinks weights enough for the largest KV pool of the three (**~9.45M tokens => C=8–9 at 1M context**, via GMU 0.83 + `--max-num-batched-tokens` 2048 to free activation reserve into the pool). Chosen serving config.
- **EXL3 3.5bpw:** fastest single-stream and smallest weights; a uniform Pollard pack. Lower fidelity to native than TR3 on the battery. Excellent when raw throughput and footprint matter most.
- **Native MXFP4:** the release's own format, the quality reference, but the heavy weights leave only a ~1.4M-token KV pool => far less concurrency/context headroom on 128 GB/node.

## Recipe

Engine image is the upstream vLLM `dsv41-feat` tree **pinned at commit `e47aa780b`** (the tree the patches target) plus the cuda-exl3 / B12X plugin layers; see [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) for the image build and the seven patches. The TR3-Hybrid plugin (B12X K3 trellis tail + MXFP4 keep tier, expert-parallel inside TP) applies the hybrid quant at load.

Deployment knobs (per rank): `TP=4, GMU 0.80, max-model-len 1048576, max-num-seqs 8, CUDA graphs FULL_AND_PIECEWISE, DSpark k=5, Engram disk-backed, fast_math on, decode block_m 8 / prefill block_m 64`.

## Speed-knob tuning

The deployed config is `fast_math on, decode block_m 8, prefill block_m 64`. Measured C=1 decode tok/s across three TR3 settings (all same weights):

| category | TR3 baseline (bm8, no fast_math) | TR3 bm32 + fast_math | **TR3 bm8 + fast_math (CURRENT SERVE)** |
|---|---|---|---|
| prose | 23.8 | 27.7 | **26.0** |
| list | 36.0 | 37.9 | **42.0** |
| code | 51.9 | 38.9 | **46.6** |
| essay | 25.3 | 25.5 | **29.9** |
| read | 37.6 | 38.7 | **35.2** |
| math | 49.3 | 43.3 | **48.3** |

`block_m 8 + fast_math` is the best balance: it wins prose/list/essay, matches math, and keeps code near baseline, whereas `block_m 32` gained prose but lost ~25% on code. Deployed.

## Credits

This work stands on other people's. Please credit them:

- **[brandonmusic](https://huggingface.co/brandonmusic)** — the **TR3 hybrid quant method** (native-keep hardest experts + EXL3-TR3 K3 tail), first published as [`brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid`](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid), and EXL3 work. The weights here port his method onto DeepSeek-V4.1-Flash.
- **WestWaters** — the **Pollard** quantization method used by the EXL3 comparison pack.
- **[tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)** — the four-DGX-Spark vLLM serving recipe, the seven patches (Engram rank offsets, CUDA-graph staging, SM120 sparse-attention fixes, streaming loader), the overlay image build chain, and the pinned-commit diagnosis that made graphs work.
- **[bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)** — the EXL3 3.5 bpw Pollard checkpoint and the V4.1 cuda-exl3 config/moe files.
- **[Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3)** — the EXL3 CUDA plugin and kernels.
- **Luke Alonso** — the TP3 virtual-heads idea (from MiniMax-M3).
- **The [vLLM](https://github.com/vllm-project/vllm) project** — DeepSeek-V4.1-Flash support and the engine.
- **[DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** — the base model.

## License

MIT for this repo's scripts and notes. The model weights follow their upstream licenses (see the HuggingFace weights repo and DeepSeek-AI's license).
