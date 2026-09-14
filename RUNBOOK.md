# DeepSeek-V4.1-Flash native vLLM on four DGX Sparks (.1/.2/.3/.5) — runbook

## STANDING (2026-09-14) — TR3-Hybrid 1M ctx, 8M+ KV pool (9,452,923 tokens)
Champion serving config, live on .1:8000. Pool grown from ~4.5M to 9.45M by the activation-reserve lever:
```
IMAGE=vllm-dsv41:overlay5-e47aa PATCH_SET=patch-upstream-boot10 TR3=1 CACHE_TAG=tr3 \
  SERVED_NAME=deepseek-v4.1-flash EAGER=0 SPEC=1 SPEC_K=5 TEXT_ONLY=0 EXTRA=0 \
  GMU=0.83 MAXLEN=1048576 SEQS=8 BATCH=2048 \
  TR3_FAST_MATH=1 TR3_DECODE_BLOCK_M=8 TR3_PREFILL_BLOCK_M=64 \
  CUDAGRAPH_MODE=FULL_AND_PIECEWISE KERNEL_CONFIG='{"enable_flashinfer_autotune": false}' \
  python3 cluster.py start
```
- KV pool 9,452,923 tokens (9.02x a full 1M request), available KV 29.69 GiB/rank. Vision+tools+graphs+DSpark k=5 all on.
- LEVER: vLLM reserves prefill activation ~ --max-num-batched-tokens. BATCH 8192->2048 + GMU 0.80->0.83 freed ~14 GiB of activation reserve into the pool (KV 15.89->29.69 GiB). BATCH floor is 2048 with vision (mm-item=1025); TEXT_ONLY=1 allows 1025.
- Ceiling: fp8_ds_mla is the KV-dtype floor for V4.1 (~3.14 KB/token/rank), so ~10.2M with vision at GMU 0.85, ~12M text-only; 12M not reachable at 1M with vision at safe GMU.
- C1->C8 sweep (short prompts, 256 tok): per-stream C1 math 44.6/code 37.3/prose 22.2; aggregate C8 math 171.3/list 135.8/code 130.2/prose 77.8.


Serving as of 2026-09-12 10:08 EDT. Endpoint http://10.100.10.1:8000, model `deepseek-v4.1-flash`,
OpenAI chat/completions, vision (up to 4 images), tool calling (`deepseek_v41` parsers), DSpark k=5.




## 2026-09-12 19:27 EDT — EXL3 3.5 bpw TP4 lane (upstream exl3tp4b) VERIFIED

Checkpoint `.3:/home/keyspark/models/DeepSeek-V4.1-Flash-EXL3-Pollard` = bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard
shards 3-42 (sha256-verified against upstream `exl3/tools/exl3_body_sha256.txt`) + shards 1,2,43-48 and small files
hardlinked from the release dir. Image `vllm-dsv41:exl3a-e47aa` = overlay5-e47aa + Zeuss5/cuda-exl3 6a1ffc34
(build it on EACH node from its local base: `docker load` of a foreign build does not share layers, 24-48 GB).
Patches `patch-upstream-exl3-tp3e/` (= upstream `patch/exl3-tp3`, 14 files, MD5SUMS checked).
```
IMAGE=vllm-dsv41:exl3a-e47aa PATCH_SET=patch-upstream-exl3-tp3e EXTRA=0 CACHE_TAG=exl3a \
MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-Pollard EAGER=0 SPEC=1 TEXT_ONLY=0 ENGRAM_LOCAL=1 \
GMU=0.80 MAXLEN=300000 SEQS=8 BATCH=8192 python3 cluster.py start
```
Result: model 56.6 GiB/rank (native 81.6), KV pool 3,445,521 tokens (11.5 x 300K), functest ALL PASS, DSpark accept 0.76.
At MAXLEN=1048576: KV pool 3,505,077 tokens (3.34x); 1M needle PASS (prompt 1,047,341 tokens, depth 0.5, TTFT 1448 s ~723 tok/s). needle.py: real_tokens ~= 1.20 x TOK here, so `needle.py BASE 866000 0.5` for a ~1.04M-token run.

### Bench boot42 EXL3 lane, first run after boot (19:28-19:43 EDT)

### Per-stream decode tok/s (after first token)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 31.7 | 13.5 | 13.3 | 8.5 |
| code | 56.5 | 26.7 | 30.4 | 18.5 |
| list | 47.9 | 25.0 | 26.8 | 18.6 |
| json | 68.0 | 38.3 | 37.9 | 19.5 |
| math | 39.7 | 28.3 | 22.8 | 17.2 |

### Aggregate tok/s (wall, TTFT included)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 30.9 | 26.2 | 50.2 | 65.7 |
| code | 54.1 | 50.7 | 105.8 | 128.3 |
| list | 45.7 | 48.0 | 100.9 | 134.6 |
| json | 64.3 | 70.2 | 132.1 | 142.7 |
| math | 37.9 | 54.0 | 82.7 | 113.1 |

### TTFT s

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 0.22 | 0.44 | 0.49 | 0.66 |
| code | 0.2 | 0.4 | 0.4 | 0.49 |
| list | 0.25 | 0.36 | 0.26 | 0.58 |
| json | 0.22 | 0.42 | 0.43 | 0.8 |
| math | 0.3 | 0.46 | 0.72 | 2.7 |

vs boot39 native pinned C1: prose 27.8 / code 39.2 / list 35.5 / json 59.3 / math 56.6.

### Bench boot42 EXL3 lane, repeat run 20 min later (19:45-20:00 EDT)

### Per-stream decode tok/s (after first token)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 25.8 | 17.6 | 10.6 | 9.8 |
| code | 34.2 | 42.7 | 19.6 | 14.0 |
| list | 31.7 | 40.7 | 29.6 | 18.7 |
| json | 47.8 | 65.1 | 44.5 | 32.6 |
| math | 42.3 | 47.1 | 38.0 | 26.2 |

### Aggregate tok/s (wall, TTFT included)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 25.0 | 33.9 | 39.9 | 71.9 |
| code | 33.1 | 78.6 | 73.3 | 102.3 |
| list | 30.4 | 77.2 | 111.8 | 140.4 |
| json | 45.4 | 120.2 | 163.5 | 241.0 |
| math | 39.6 | 84.0 | 137.2 | 192.7 |

### TTFT s

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 0.31 | 0.47 | 0.57 | 0.56 |
| code | 0.26 | 0.24 | 0.43 | 0.71 |
| list | 0.35 | 0.27 | 0.29 | 0.51 |
| json | 0.28 | 0.29 | 0.43 | 0.41 |
| math | 0.42 | 0.34 | 0.38 | 0.45 |

Run-to-run variance is large: C1 fell (prose 25.8, code 34.2) while C8 aggregate rose (json 241, math 193, list 140). Take C1 ~26-32 prose / 34-57 code and C8 aggregate 100-240 as the band.

## 2026-09-12 18:10 EDT — ROOT CAUSE (upstream) and the replication recipe

tonyd2wild's docs/RECIPE.md ("Pin the branch commit, not the branch"): the seven patch files target vLLM `dsv41-feat`
at commit `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`. The branch was force-pushed on 2026-09-11 00:49 UTC; an overlay
built from the branch head after that plus these patches "boots without a single error, passes profiling and graph
capture, and emits one repeated garbage token from the first position, with DSpark accepting nothing". Our
`vllm-dsv41:overlay1` was built 2026-09-11 03:30 UTC (after the force-push) — that is the tree every native boot here
ran on. Fix: the pinned tree. `/home/keyspark/vllm-e47aa` = vLLM at e47aa780b; `vllm-dsv41:overlay5-e47aa` on all four
nodes = `FROM overlay5` + `COPY vllm/` of that tree (upstream's Dockerfile.overlay step on top of the compiled layers).
Patches: `patch-upstream-boot10/` (byte-identical to upstream `patch/`), no patch-extra.
```
IMAGE=vllm-dsv41:overlay5-e47aa PATCH_SET=patch-upstream-boot10 EXTRA=0 CACHE_TAG=e47aa \
EAGER=0 SPEC=1 TEXT_ONLY=0 ENGRAM_LOCAL=1 GMU=0.80 MAXLEN=300000 SEQS=8 BATCH=8192 python3 cluster.py start
```
(= upstream boot10 knobs; first boot recompiles DeepGEMM/autotune into /cache/vllm-e47aa, ~10 min extra.)
**VERIFIED 2026-09-12 18:55 EDT (boot39): functest ALL PASS, DSpark accept 0.81, autotune ON. This is the standing native recipe now.**
Upstream's default lane since 2026-09-11 is EXL3 3.5 bpw TP4 (`exl3tp4b`, bot-lab-21 Pollard experts, KV pool 3.3M
tokens, +8-25% throughput over boot10); `/home/keyspark/models/DSV4.1-Flash-EXL3-TP4` on .3 holds a 20 GB EXL3 export.
Before trusting any bench: 10 s fp16 burn per node (healthy 75-90 TFLOPS, 2.2-2.4 GHz, 80 W+; the GB10 EC clock
latch after a reboot shows only as speed; fix = unplug the adapter 30-60 s). `tools/prelaunch-quick.sh` upstream.


### Bench boot38 (graphs + DSpark + vision, 1M, autotune OFF; 18:15-18:30 EDT)

Correct output but slower than eager boot33 (prose 26.6) and boot8 (prose 29.7): FlashInfer autotune + sparse-MLA calibration is worth ~30% when it works.

### Per-stream decode tok/s (after first token)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 20.0 | 14.3 | 11.7 | 9.3 |
| code | 30.9 | 25.9 | 20.7 | 14.7 |
| list | 28.0 | 21.5 | 19.9 | 15.4 |
| json | 39.7 | 40.7 | 32.2 | 23.1 |
| math | 34.0 | 32.4 | 22.7 | 16.2 |

### Aggregate tok/s (wall, TTFT included)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 19.6 | 26.5 | 44.4 | 70.8 |
| code | 29.5 | 49.2 | 74.1 | 110.4 |
| list | 26.9 | 41.4 | 74.8 | 116.4 |
| json | 37.8 | 76.1 | 116.0 | 168.1 |
| math | 32.3 | 59.3 | 86.0 | 115.3 |

### TTFT s

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 0.27 | 0.22 | 0.51 | 0.58 |
| code | 0.4 | 0.41 | 0.59 | 0.42 |
| list | 0.4 | 0.32 | 0.37 | 0.62 |
| json | 0.32 | 0.43 | 0.39 | 0.72 |
| math | 0.39 | 0.43 | 0.45 | 1.37 |


### Bench boot39 = STANDING (upstream pinned tree, boot10 knobs, 300K, autotune ON; 18:58-19:13 EDT)

### Per-stream decode tok/s (after first token)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 27.8 | 20.6 | 14.6 | 9.8 |
| code | 39.2 | 35.9 | 24.2 | 15.6 |
| list | 35.5 | 29.5 | 23.3 | 16.6 |
| json | 59.3 | 46.2 | 32.6 | 25.5 |
| math | 56.6 | 40.7 | 27.4 | 20.4 |

### Aggregate tok/s (wall, TTFT included)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 26.9 | 39.4 | 56.1 | 73.3 |
| code | 37.6 | 67.0 | 89.6 | 113.4 |
| list | 33.8 | 56.5 | 86.5 | 123.5 |
| json | 55.6 | 84.8 | 117.4 | 188.9 |
| math | 52.8 | 73.8 | 101.3 | 132.5 |

### TTFT s

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 0.3 | 0.22 | 0.47 | 0.59 |
| code | 0.27 | 0.22 | 0.33 | 0.61 |
| list | 0.37 | 0.31 | 0.33 | 0.42 |
| json | 0.29 | 0.41 | 0.48 | 0.41 |
| math | 0.32 | 0.45 | 0.44 | 2.44 |

## 2026-09-12 16:00 EDT — INTERIM CONFIG (eager + DSpark) after the 10:20 cluster reboot

All four nodes hard-rebooted at 10:20 EDT (TR3 boot9 host-memory blowup + panic_on_oom). Since then the
boot8 recipe with CUDA graphs returns NaN logits for every request (boots 31/32; "�carecare..." output,
DSpark accept 0). The identical recipe with `EAGER=1` is correct (boot33: functest ALL PASS, DSpark accept
0.82). Kernel/driver/packages/patches/caches/mounts verified unchanged; root cause of the graph-capture
poisoning is open (see HANDOFF.md). Interim launch line:
```
EAGER=1 SPEC=1 TEXT_ONLY=0 ENGRAM_LOCAL=1 EXTRA=1 GMU=0.83 MAXLEN=1048576 SEQS=8 BATCH=4096 python3 cluster.py start
```
Memory guard: do NOT run memguard.sh below 3 GiB on this recipe (native load dips to ~12 GiB, the 1M/GMU 0.83
steady state is 4-6 GiB MemAvailable, 2-3 GiB under C8 load). Boots 29/30 were killed by 12/6 GiB floors.
Bench for this config (16:03-16:20 EDT, 256-token completions):
### Per-stream decode tok/s (after first token)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 26.6 | 20.0 | 10.1 | 9.0 |
| code | 43.3 | 33.8 | 16.8 | 15.1 |
| list | 39.6 | 28.4 | 16.7 | 15.6 |
| json | 55.5 | 38.1 | 30.7 | 24.7 |
| math | 37.1 | 40.2 | 23.7 | 20.0 |

### Aggregate tok/s (wall, TTFT included)

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 25.9 | 39.1 | 37.8 | 67.1 |
| code | 41.8 | 65.0 | 59.3 | 107.2 |
| list | 38.0 | 53.2 | 62.6 | 116.4 |
| json | 52.9 | 70.1 | 110.7 | 182.5 |
| math | 35.5 | 74.5 | 87.4 | 148.8 |

### TTFT s

| class | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| prose | 0.25 | 0.24 | 0.65 | 0.54 |
| code | 0.22 | 0.28 | 0.5 | 0.68 |
| list | 0.28 | 0.3 | 0.47 | 0.43 |
| json | 0.23 | 0.46 | 0.51 | 0.47 |
| math | 0.32 | 0.45 | 0.61 | 0.55 |

boot8 (graphs, pre-reboot) reference: prose 29.7/22.6/14.4/10.4, json 63.3, math 52.5 C1; C8 aggregate 79-192.

## Operate
```
cd /home/keyspark/dsv41-recovery-20260912
python3 cluster.py deploy                                   # ships serve-rank.sh, patch/, patch-extra/, engram_local.py
KERNEL_CONFIG='{"enable_flashinfer_autotune": false}' EAGER=0 SPEC=1 TEXT_ONLY=0 ENGRAM_LOCAL=1 EXTRA=1 GMU=0.83 MAXLEN=1048576 SEQS=8 BATCH=4096 python3 cluster.py start
#   KERNEL_CONFIG added 2026-09-12 18:12 EDT (boot38): FlashInfer autotune + the SM120 sparse-MLA timing calibration
#   produced NaN logits after the 10:20 cluster reboot (graphs mode); disabling it restores correct output.
python3 cluster.py status | logs --tail 80 | stop            # stop touches only label keyspark.task=dsv41-recovery-20260912
nohup ./watch-boot.sh > bootN-watch.log &                    # READY / RANK CONTAINER exit / TIMEOUT
python3 functest.py http://10.100.10.1:8000                  # text, vision, tools, DSpark metrics (5-10 s) -- run after EVERY boot
./run-bench.sh TAG 256 ; python3 needle.py BASE TOKENS 0.5
```
Ranks start 3,2,1,0 (.5,.3,.2,.1); rank0 = .1 serves the API. Cold boot ~13 min with warm JIT caches
(`/home/keyspark/dsv41-native-cache` per node); the very first boot was 63 min (MAX_JOBS=2 keeps the
compilers from OOMing the hosts). Preflight asserts: native checkpoint, all shards, GID3 = node IP, >100 GiB free.

## Knobs (serve-rank.sh env)
GMU (<=0.85; 0.83 needed for 1M+8 seqs), MAXLEN, SEQS, BATCH (4096 keeps the FlashInfer autotune cache
valid), EAGER (1 = --enforce-eager), SPEC (DSpark on/off) + SPEC_K, TEXT_ONLY (--language-model-only),
ENGRAM_LOCAL (node-local sparse Engram rows on .1/.5; .2/.3 read the source), EXTRA (mount patch-extra/).
Graph sizes: k*i and (k+1)*i for i<=SEQS with DSpark, 1..SEQS without (padded batches hang SM120 sparse MLA).

## Patch sets
* `patch/` (v2): tonyd2wild's seven fixes RE-DERIVED from this image's own sources (imgsrc/, patchwork/).
  Their byte-for-byte boot10 files (patch-upstream-boot10/) belong to the pre-merge dsv41-feat tree and
  silently downgraded weight_utils.py / sparse_attn_indexer.py / sparse_swa.py on our merged-tree image.
* `patch-extra/`: vLLM PR #56598 (DSpark padded draft rows bypass the dsv4 fast top-k router) and
  PR #56448 (cap DFlash/DSpark profiling query batch). REQUIRED with DSpark + CUDA graphs: without it every
  response is NaN garbage ("�carecare...") because out-of-range expert ids corrupt GPU memory.
  Re-check when the image is rebuilt from a tree that already contains them.

## Verified (boot8, 1M ctx, 8 seqs, GMU .83, patch v2 + extra)
* functest ALL PASS: 17*23=391; Canberra/1969; vision OCR "SPARK 7731" + red circle + blue rectangle;
  get_weather(Tokyo, celsius) tool call parsed; DSpark accepted 128/165 draft tokens (77.6%, 3.88 per draft).
* KV pool 2,465,192 tokens (2.35 x 1M). Graph memory 0.75 GiB (rank0). Init engine 276 s.
* Throughput (256 output tokens, temperature 0.7, usage.completion_tokens counted, streaming TTFT):

| class | decode tok/s per stream C1/C2/C4/C8 | aggregate tok/s C1/C2/C4/C8 | TTFT s C1 |
|---|---|---|---|
| prose | 29.7 / 22.6 / 14.4 / 10.4 | 28.7 / 42.7 / 53.9 / 79.0 | 0.30 |
| code | 33.7 / 33.1 / 20.9 / 17.1 | 32.5 / 60.8 / 76.5 / 123.4 | 0.28 |
| list | 39.9 / 26.8 / 22.7 / 16.8 | 37.7 / 51.5 / 86.5 / 126.9 | 0.37 |
| json | 63.3 / 50.0 / 37.2 / 26.2 | 58.2 / 92.2 / 136.0 / 191.9 | 0.35 |
| math | 52.5 / 47.0 / 28.5 / 21.6 | 48.5 / 84.0 / 103.8 / 159.4 | 0.40 |

  Reference points: eager no-spec 15 tok/s; graphs no-spec 29; eager+DSpark prose 22 / code 39 / list 36.
  The user's 50-80 tok/s expectation holds for json/math single-stream and for aggregate at C2+; prose/code
  single-stream sit at 30-34 (DSpark acceptance is lower on free text), consistent with upstream's boot10.
* Needle-in-haystack (code at depth 0.5, greedy, DSpark on, needle.py):

| target | prompt tokens | TTFT s | prefill tok/s | result |
|---|---|---|---|---|
| 32K (eager boot4) | 37,778 | 32.5 | 1,160 | PASS |
| 128K | 153,974 | 127.7 | 1,206 | PASS |
| 300K | 362,194 | 263.2 | 1,376 | PASS |
| 600K | 725,341 | 575.0 | 1,261 | PASS |
| ~1M (target 850K) | 1,027,974 | 974.1 | 1,055 | PASS |

## TR3 / B12X port status (2026-09-12 afternoon)
Single-expert K3 tail kernel validated on GB10 (see HANDOFF). Plugin port in progress under hybrid-port/.

## Limitations / not done
* Custom TR3-Hybrid (EXL3 K3 tail + native keep experts) is NOT deployed. Brandon B12X v84 kernel refused
  V4.1's swiglu_limit=10 with intermediate rotation; patched copy hybrid-port/b12x (clamp on the un-rotated
  gate/up inside the rotation epilogue, diff hybrid-port/b12x-swiglu-clamp-rotation.diff) is UNTESTED —
  run-b12x-tail.sh needs a free GPU on .3 (all four are serving). The full vLLM plugin port (EP-in-TP, keep
  tier, rank ownership, draft path) is still ahead of that. Do not present the native serve as the TR3 one.
* 1M-wide DSpark verification/top-k was only validated upstream to 300K rows; the needle series here (PASS at
  362K, 725K and 1.028M tokens, server healthy afterwards) is the first evidence beyond that. Prefill is
  ~1.0-1.4K tok/s, so a full-1M prompt costs ~16 min TTFT.
* Node .2 has 19 GB free disk: no node-local Engram copy there (reads .3 over NFS). Host memory headroom at
  GMU .83 is ~5 GiB per node during warmup; do not raise GMU further.
