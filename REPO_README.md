# DeepSeek-V4.1-Flash TR3-Hybrid — 1M Context, 8M-token KV pool, Four DGX Sparks, with abliterated option

Serving **[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** (763B, MoE, sparse-MLA + Engram) on **four NVIDIA DGX Spark (GB10, 128 GB unified)** over a 200G ConnectX-7 fabric, tensor-parallel 4, with **CUDA graphs + DSpark speculative decoding + native vision + tool calling + 1M-token context**.

The serving checkpoint is the **TR3-Hybrid** quant: most routed experts are **EXL3-TR3 K=3 (mcg codebook, ~3.0 bpw)**, the 64 highest round-trip-error experts per layer stay **native MXFP4**, everything else (attention, indexer, Engram tables, hyper-connections) is the release's own format. ~3.22 bpw MoE mix.

> **Weights:** https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid

## Headline

- **1M-token context**, KV pool **9,452,923 tokens (8M+ target)** => **~9 concurrent full-1M requests (C=8-9)** on four Sparks.
- Model weights **69.23 GiB/rank** (vs 81.6 GiB/rank for the native release pack), which is what frees the KV room.
- **1M needle-in-haystack: PASS** (retrieval at depth 0.5 in a >1M-token prompt).
- Quality matches the full-precision model on an objective battery; closer to native than the uniform EXL3 3.5 bpw pack (below).

## Results

All speed numbers are greedy, temperature 0, measured on this cluster. Native / EXL3 / TR3 rows are matched at 300K context, GMU 0.80; the optimized TR3 column is the 1M-context deployment.

### Single-stream decode tok/s (C=1)

| category | Native MXFP4 | TR3-Hybrid | EXL3 3.5bpw | TR3-Hybrid (optimized, 1M) |
|---|---|---|---|---|
| prose | 27.4 | 23.8 | 30.7 | 27.7 |
| list | 41.8 | 36.0 | 43.7 | 37.9 |
| code | 49.0 | 51.9 | 58.2 | 38.9 |
| essay | 28.9 | 25.3 | 34.4 | 25.5 |
| read | 37.3 | 37.6 | 41.6 | 38.7 |
| math | 56.7 | 49.3 | 64.0 | 43.3 |

### Aggregate throughput tok/s (C=8, wall-clock incl. TTFT)

| category | Native MXFP4 | TR3-Hybrid | EXL3 3.5bpw | TR3-Hybrid (optimized, 1M) |
|---|---|---|---|---|
| prose | 53.6 | 49.9 | 68.3 | 50.5 |
| list | 70.6 | 91.3 | 111.0 | 90.4 |
| code | 84.1 | 95.9 | 110.1 | 94.5 |
| essay | 57.2 | 60.9 | 78.1 | 49.0 |
| read | 63.8 | 77.4 | 74.5 | 63.1 |
| math | 84.4 | 93.6 | 137.2 | 103.3 |

### Aggregate throughput tok/s (C=8)

| category | Native MXFP4 | TR3-Hybrid | EXL3 3.5bpw | TR3-Hybrid (optimized, 1M) |
|---|---|---|---|---|
| prose | 73.3 | 79.8 | 88.1 | 70.3 |
| list | 117.0 | 127.8 | 149.4 | 140.0 |
| code | 120.2 | 136.1 | 157.2 | 129.2 |
| essay | 77.7 | 87.5 | 102.2 | 76.2 |
| read | 109.9 | 132.1 | 123.5 | 114.7 |
| math | 153.4 | 173.4 | 189.2 | 176.5 |

### Memory, KV pool, context

| config | weights GiB/rank | KV pool (tokens) | measured at | max context |
|---|---|---|---|---|
| Native MXFP4 | 81.58 | 1,428,283 | 300K | 1M |
| EXL3 3.5bpw | 56.61 | 3,534,988 | 300K | 1M |
| TR3-Hybrid | 69.21 | 4,190,217 | 300K | 1M |
| **TR3-Hybrid (deployed, 8M-pool)** | 69.2 | 9,452,923 | **1M** | **1M** |

### Intelligence / quality (30-item battery, greedy; native = full-precision reference)

| metric | Native | TR3-Hybrid | EXL3 3.5bpw |
|---|---|---|---|
| objective correct | 24/27 | 25/27 | 25/27 |
| top-1 token agreement vs native | 1.000 | 0.603 | 0.466 |

All three are tied on task correctness. TR3-Hybrid tracks the full-precision model **closer** than EXL3 3.5 bpw (higher top-1 agreement), consistent with the design intent of keeping the hardest experts in native precision.

## Why each config

- **TR3-Hybrid (deployed):** best precision-for-memory on this fabric. Keeps the 64 hardest experts/layer native so quality tracks the full model, while the K3 trellis tail shrinks weights enough for the largest KV pool of the three (~9.45M tokens => C=8-9 at 1M context, via GMU 0.83 + max-num-batched-tokens 2048). Chosen serving config.
- **EXL3 3.5bpw:** fastest single-stream and smallest weights; a uniform Pollard pack. Lower fidelity to native than TR3 on the battery. Excellent when raw throughput and footprint matter most.
- **Native MXFP4:** the release's own format, the quality reference, but the heavy weights leave only a ~1.4M-token KV pool => far less concurrency/context headroom on 128 GB/node.

## Recipe

Engine image is the upstream vLLM `dsv41-feat` tree **pinned at commit `e47aa780b`** (the tree the patches target) plus the cuda-exl3 / B12X plugin layers; see [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) for the image build and the seven patches. The TR3-Hybrid plugin (B12X K3 trellis tail + MXFP4 keep tier, expert-parallel inside TP) applies the hybrid quant at load.

Deployment knobs (per rank): `TP=4, GMU 0.80, max-model-len 1048576, max-num-seqs 8, CUDA graphs FULL_AND_PIECEWISE, DSpark k=5, Engram disk-backed, fast_math on, decode block_m 32 / prefill block_m 64`.

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
