#!/usr/bin/env bash
# Native DeepSeek-V4.1-Flash TP4 with disk-backed Engram.
# Adapted from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (MIT).
set -euo pipefail
RANK="${1:?rank required}"
IPS=(10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5)
[[ "$RANK" =~ ^[0-3]$ ]] || exit 2
IP="${IPS[$RANK]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=dsv41-native
IMAGE="${IMAGE:-vllm-dsv41:overlay5}"   # overlay5-e47aa = pinned dsv41-feat e47aa780b tree (upstream recipe)
TR3="${TR3:-0}"
TR3_NATIVE="${TR3_NATIVE:-0}"   # 1 = run the NATIVE checkpoint through the TR3 plugin (bisect: keep tier only)
MODEL=DeepSeek-V4.1-Flash
[[ "$TR3" != 1 || "$TR3_NATIVE" == 1 ]] || MODEL=DeepSeek-V4.1-Flash-TR3-Hybrid
[[ -z "${MODEL_DIR:-}" ]] || MODEL="$MODEL_DIR"   # e.g. DeepSeek-V4.1-Flash-EXL3-Pollard (upstream exl3tp4b lane)
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash}"
SOURCE="/mnt/models-node2/$MODEL"
[[ "$RANK" != 2 ]] || SOURCE="/home/keyspark/models/$MODEL"
CACHE=/home/keyspark/dsv41-native-cache
ENGRAM=/home/keyspark/dsv41-engram-local
GMU="${GMU:-0.80}"
MAXLEN="${MAXLEN:-131072}"
SEQS="${SEQS:-4}"
BATCH="${BATCH:-4096}"
EAGER="${EAGER:-0}"
SPEC="${SPEC:-1}"
TEXT_ONLY="${TEXT_ONLY:-0}"
ENGRAM_LOCAL="${ENGRAM_LOCAL:-1}"
SPEC_K="${SPEC_K:-5}"
python3 - "$GMU" "$SOURCE" "$IP" "$TR3" "$TR3_NATIVE" <<'PY'
import ipaddress,json,sys
from pathlib import Path
gmu,model,ip,tr3,tr3_native=sys.argv[1:]
assert 0 < float(gmu) <= .85, 'GMU must be <= 0.85'
p=Path(model)
d=json.loads((p/'model.safetensors.index.json').read_text())
missing=[s for s in set(d['weight_map'].values()) if not (p/s).is_file()]
assert not missing, missing
cfg=json.loads((p/'config.json').read_text())
assert ('hybrid_tr3_tail' in cfg) == (tr3 == '1' and tr3_native != '1'), ('TR3 knob does not match checkpoint', tr3, tr3_native, 'hybrid_tr3_tail' in cfg)
gid=Path('/sys/class/infiniband/rocep1s0f1/ports/1/gids/3').read_text().strip()
assert ipaddress.IPv6Address(gid).ipv4_mapped == ipaddress.IPv4Address(ip), gid
avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
assert avail > 100*1024**2, f'Only {avail/1024**2:.1f} GiB available'
print(f'Preflight rank IP={ip}: {len(set(d["weight_map"].values()))} shards, GID3 verified, {avail/1024**2:.1f} GiB available')
PY
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "$NAME already exists; inspect it or use cluster.py stop first" >&2
  exit 2
fi
mkdir -p "$CACHE" "$ENGRAM"
SITE=/usr/local/lib/python3.12/dist-packages/vllm
MOUNTS=()
PATCH_SET="${PATCH_SET:-patch}"   # patch = re-derived v2 (post-force-push tree); patch-upstream-boot10 = tonyd2wild's seven files byte-for-byte (pinned e47aa780b tree)
while read -r file relative; do
  [[ -n "$file" ]] || continue
  test -f "$HERE/$PATCH_SET/$file"
  case "$relative" in /*) tgt="$relative" ;; *) tgt="$SITE/$relative" ;; esac   # exl3 set mounts cuda_exl3/ files by absolute path
  MOUNTS+=(-v "$HERE/$PATCH_SET/$file:$tgt:ro")
done < "$HERE/$PATCH_SET/mounts.txt"
EXTRA="${EXTRA:-0}"
if [[ "$EXTRA" == 1 ]]; then
  while read -r file relative; do
    [[ -n "$file" ]] || continue
    test -f "$HERE/patch-extra/$file"
    MOUNTS+=(-v "$HERE/patch-extra/$file:$SITE/$relative:ro")
  done < "$HERE/patch-extra/mounts.txt"
fi
CG="$(python3 - "$SEQS" "$SPEC" "$SPEC_K" <<'PY'
import sys
n,spec,k=int(sys.argv[1]),sys.argv[2],int(sys.argv[3])
# DSpark k: target decode batches are (k+1)*i tokens, draft batches k*i; no spec: 1..n
sizes={j*i for j in ([k,k+1] if spec=="1" else [1]) for i in range(1,n+1)}
print(','.join(map(str,sorted(sizes))))
PY
)"
ARGS=(--served-model-name "$SERVED_NAME" --host 0.0.0.0 --port 8000
  --tensor-parallel-size 4 --distributed-executor-backend mp --nnodes 4
  --node-rank "$RANK" --master-addr 10.100.10.1 --master-port 29581
  --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH" --block-size 128
  --engram-config '{"cpu_offload":false}'
  --default-chat-template-kwargs '{"thinking":false}'
  --enable-auto-tool-choice --tool-call-parser deepseek_v41
  --reasoning-parser deepseek_v41)
if [[ "$TEXT_ONLY" == 1 ]]; then ARGS+=(--language-model-only); else ARGS+=(--limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1); fi
if [[ "$SPEC" == 1 ]]; then ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}"); fi
GRAPH_ENV=()
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
COMPILE_LEVEL="${COMPILE_LEVEL:-}"   # e.g. 0 = CUDA graphs without torch.compile/Inductor
LEVEL_JSON=""; [[ -z "$COMPILE_LEVEL" ]] || LEVEL_JSON="\"mode\":$COMPILE_LEVEL,"
if [[ "$EAGER" == 1 ]]; then ARGS+=(--enforce-eager); else ARGS+=(--compilation-config "{${LEVEL_JSON}\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG]}"); [[ -z "${BREAKABLE:-}" ]] || GRAPH_ENV=(-e "VLLM_USE_BREAKABLE_CUDAGRAPH=$BREAKABLE"); fi   # boot8 recipe = env UNSET (vLLM auto-enables it for DeepseekV41 after config init); explicit =1 at import time changes the break-point set and gave NaN (boot31)
[[ "$TR3" != 1 || "$TR3_NATIVE" == 1 ]] || ENGRAM_LOCAL=0   # node-local Engram copies carry the NATIVE shard layout
ENGRAM_MOUNT="$ENGRAM"; [[ "$ENGRAM_LOCAL" == 1 ]] || { ENGRAM_MOUNT=/home/keyspark/dsv41-engram-empty; mkdir -p "$ENGRAM_MOUNT"; }
TR3_ENV=(); [[ "$TR3" != 1 ]] || TR3_ENV=(-v "$HERE/hybrid-port:/brandon:ro" -e PYTHONPATH=/brandon:/brandon/plugin-root -e TR3_HYBRID=1 -e "TR3_MODEL_DIR=/models/$MODEL" -e "TR3_DEBUG=${TR3_DEBUG:-0}" -e "TR3_NO_DECODE_PLAN=${TR3_NO_DECODE_PLAN:-0}" -e "TR3_PREWARM=${TR3_PREWARM:-1}" -e "TR3_SKIP_IN_CAPTURE=${TR3_SKIP_IN_CAPTURE:-0}" -e "TR3_FAST_MATH=${TR3_FAST_MATH:-0}" -e "TR3_DECODE_BLOCK_M=${TR3_DECODE_BLOCK_M:-}" -e "TR3_PREFILL_BLOCK_M=${TR3_PREFILL_BLOCK_M:-}" -e "B12X_DYNAMIC_TILE_MN=${B12X_DYNAMIC_TILE_MN:-}")
ABLIT_CAPTURE="${ABLIT_CAPTURE:-0}"
if [[ "$ABLIT_CAPTURE" == 1 ]]; then
  ABLIT_HOST_DIR="${ABLIT_HOST_DIR:-$HERE/ablit-work}"
  mkdir -p "$ABLIT_HOST_DIR"
  TR3_ENV+=(-v "$ABLIT_HOST_DIR:/ablit_capture" -e ABLIT_CAPTURE=1 -e ABLIT_CAPTURE_DIR=/ablit_capture -e "ABLIT_CAPTURE_TAG=${ABLIT_CAPTURE_TAG:-run}" -e ABLIT_CAPTURE_TAG_FILE=/ablit_capture/CURRENT_TAG)
fi
[[ -z "${KERNEL_CONFIG:-}" ]] || ARGS+=(--kernel-config "$KERNEL_CONFIG")   # e.g. '{"enable_flashinfer_autotune": false}' skips FlashInfer autotune + SM120 sparse-MLA cpb calibration
[[ "$RANK" == 0 ]] || ARGS+=(--headless)
docker run -d --name "$NAME" --label keyspark.task=dsv41-recovery-20260912 \
  --gpus all --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 --restart no --log-opt max-size=50m --log-opt max-file=3 \
  -v "$SOURCE:/models/$MODEL:ro" -v "$CACHE:/cache" -v "$ENGRAM_MOUNT:/engram-local:ro" \
  "${MOUNTS[@]}" "${TR3_ENV[@]}" \
  -e VLLM_HOST_IP="$IP" -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "CUDA_EXL3_MODEL_PATH=/models/$MODEL" -e VLLM_CACHE_ROOT=/cache/vllm${CACHE_TAG:+-$CACHE_TAG} -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
  -e DSV41_ENGRAM_DIR=/engram-local "${GRAPH_ENV[@]}" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=10.100.10.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e TP_SOCKET_IFNAME=enp1s0f1np1 -e MN_IF_NAME=enp1s0f1np1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" "/models/$MODEL" "${ARGS[@]}"
echo "Started rank=$RANK ip=$IP context=$MAXLEN sequences=$SEQS gmu=$GMU eager=$EAGER spec=$SPEC k=$SPEC_K text_only=$TEXT_ONLY engram_local=$ENGRAM_LOCAL graphs_mode=$CUDAGRAPH_MODE compile_level=${COMPILE_LEVEL:-default} kernel_config=${KERNEL_CONFIG:-default} image=$IMAGE patch_set=$PATCH_SET cache_tag=${CACHE_TAG:-} extra=$EXTRA tr3=$TR3 tr3_native=$TR3_NATIVE model=$MODEL graphs=$CG"
