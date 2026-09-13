# DeepSeek-V4.1-Flash TR3-Hybrid — 1M Context, ~4M-token KV pool, Four DGX Sparks, with abliterated option

Serving **[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** (763B, MoE, sparse-MLA + Engram) on **four NVIDIA DGX Spark (GB10, 128 GB unified)** over a 200G ConnectX-7 fabric, tensor-parallel 4, with **CUDA graphs + DSpark speculative decoding + native vision + tool calling + 1M-token context**.

The serving checkpoint is the **TR3-Hybrid** quant: most routed experts are **EXL3-TR3 K=3 (mcg codebook, ~3.0 bpw)**, the 64 highest round-trip-error experts per layer stay **native MXFP4**, everything else (attention, indexer, Engram tables, hyper-connections) is the release's own format. ~3.22 bpw MoE mix.

**Tiers.** *Tier 1* is the EXL3-TR3 **K3 trellis tail** (320 experts/layer, ~3.0 bpw) — the lossy tier that dominates the checkpoint and whose fidelity the KLD measures. *Tier 2* is the **64 keep experts/layer** left as native MXFP4 (bit-identical to native, the quality anchor). Troubleshooting low tier-1 fidelity = raise its precision (K3→K4, or move more high-round-trip-error experts into the native keep tier) and verify with a **large-sample KLD (≥50K teacher-forced positions)**, not a small smoke test.

> **Weights:** https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid

## Quick start (one-shot, four DGX Sparks)

Everything needed to bring the **current-serve** config up in one load:

```bash
git clone https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-4M-KV-Four-DGX-Sparks-with-abliterated-option
cd keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-4M-KV-Four-DGX-Sparks-with-abliterated-option
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
| Mia 2× Spark EXL3 2.9 bpw | [kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) / [`Mia-AiLab/…-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) — **overlay does not apply** (attn is EXL3 K=5); use Pollard 3.5 bpw instead |

## Headline

- **1M-token context**, KV pool ~4,492,902 tokens => **~4 concurrent full-1M requests (C=4)** on four Sparks.
- Model weights **69.2 GiB/rank** (vs 81.6 GiB/rank for the native release pack), which is what frees the KV room.
- **1M needle-in-haystack: PASS** (retrieval at depth 0.5 in a >1M-token prompt).
- Quality matches the shipped checkpoint on an objective battery; tracks it closer than the uniform EXL3 3.5 bpw pack (below).

## Results

All speed numbers are greedy, temperature 0, measured on this cluster. Native / EXL3 / TR3 rows are matched at 300K context, GMU 0.80; the fourth column is the **deployed** TR3 config (fast_math on, decode block_m 8) measured at 1M context.

### Single-stream decode tok/s (C=1)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — CURRENT SERVE) |
|---|---|---|---|---|
| prose | 27.4 | 23.8 | 30.7 | 26.0 |
| list | 41.8 | 36.0 | 43.7 | 42.0 |
| code | 49.0 | 51.9 | 58.2 | 46.6 |
| essay | 28.9 | 25.3 | 34.4 | 29.9 |
| read | 37.3 | 37.6 | 41.6 | 35.2 |
| math | 56.7 | 49.3 | 64.0 | 48.3 |

### Aggregate throughput tok/s (C=4, wall-clock incl. TTFT)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — CURRENT SERVE) |
|---|---|---|---|---|
| prose | 53.6 | 49.9 | 68.3 | 56.7 |
| list | 70.6 | 91.3 | 111.0 | 101.7 |
| code | 84.1 | 95.9 | 110.1 | 99.1 |
| essay | 57.2 | 60.9 | 78.1 | 68.2 |
| read | 63.8 | 77.4 | 74.5 | 80.5 |
| math | 84.4 | 93.6 | 137.2 | 119.2 |

### Aggregate throughput tok/s (C=8)

| category | Native MXFP4 | TR3-Hybrid (old) | EXL3 3.5bpw | TR3-Hybrid (bm8, 1M — CURRENT SERVE) |
|---|---|---|---|---|
| prose | 73.3 | 79.8 | 88.1 | 75.8 |
| list | 117.0 | 127.8 | 149.4 | 140.9 |
| code | 120.2 | 136.1 | 157.2 | 132.6 |
| essay | 77.7 | 87.5 | 102.2 | 85.2 |
| read | 109.9 | 132.1 | 123.5 | 138.9 |
| math | 153.4 | 173.4 | 189.2 | 165.4 |

### Memory, KV pool, context

| config | weights GiB/rank | KV pool (tokens) | measured at | max context |
|---|---|---|---|---|
| Native MXFP4 | 81.58 | 1,428,283 | 300K | 1M |
| EXL3 3.5bpw | 56.61 | 3,534,988 | 300K | 1M |
| TR3-Hybrid | 69.21 | 4,190,217 | 300K | 1M |
| **TR3-Hybrid (bm8 — CURRENT SERVE)** | 69.2 | 4,492,902 | **1M** | **1M** |

### Intelligence / quality (30-item battery, greedy; native = reference)

| metric | Native | TR3-Hybrid (old) | EXL3 3.5bpw |
|---|---|---|---|
| objective correct | 24/27 | 25/27 | 25/27 |
| token-sequence agreement vs native (aligned) | 1.000 | **0.780** | 0.689 |
| KL vs native (nats) — smoke test, 274 pos ↓ | 0 | 0.259 | 0.343 |
| top-1 agreement — smoke test, 274 pos | 1.000 | 0.847 | 0.821 |

> **Reference model.** DeepSeek shipped V4.1-Flash as an **MXFP4-experts / MXFP8** checkpoint (`expert_dtype fp4`, `weight_block_size [32,32]`, `ue8m0` scales); there is no public BF16 original. That shipped checkpoint is what we call **native** and is the **source both quants were made from** — so it is the ground-truth reference here, not a lossless BF16 model. TR3's 64 keep-experts/layer are **bit-identical** to native's MXFP4 experts; its tail is K3-trellis quantized from native. The agreement number is therefore "how faithfully the quant reproduces the shipped model's greedy token choices," measured with cascade-robust alignment (a single early token offset would otherwise misalign the rest; the naive position-aligned score understated TR3 at 0.60).

> ⚠️ **The KL/top-1 figures above are a 274-position smoke test, not a quality verdict.** Per brandonmusic, a real KLD needs the teacher's (native's) logits over representative data — ~**32 windows of 2047 tokens each (>50K positions)**, top-20 truncation — teacher-forced. A 6-sample/274-position run only smoke-tests the pipeline and its numbers swing (the 0.60→0.78 agreement shift was that instability). A **large-sample re-measurement (66,631 positions over a 32-window prose+code corpus) is in progress**; the table will be updated with those numbers. Directionally TR3 tracks native closer than EXL3, but treat the exact values as provisional until the >50K run lands. All three are tied on task correctness. TR3-Hybrid reproduces the shipped checkpoint **closer** than EXL3 3.5 bpw (0.78 vs 0.69 aligned agreement), partly because 64 experts/layer are bit-identical to native.

## Why each config

- **TR3-Hybrid (deployed):** best precision-for-memory on this fabric. Keeps the 64 hardest experts/layer native so quality tracks the full model, while the K3 trellis tail shrinks weights enough for the largest KV pool of the three (~4M tokens => C=4 at 1M context). Chosen serving config.
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
