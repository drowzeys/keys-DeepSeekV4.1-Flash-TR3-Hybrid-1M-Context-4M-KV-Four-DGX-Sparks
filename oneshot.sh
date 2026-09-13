#!/bin/bash
# One-shot bring-up of DeepSeek-V4.1-Flash TR3-Hybrid on four DGX Sparks (GB10), the CURRENT-SERVE config:
#   TR3-Hybrid (bm8 + fast_math), CUDA graphs + DSpark k=5 + vision + tools, 1M context, ~4.5M-token KV pool, C=4.
#
# Assumptions (this cluster's topology; edit the vars for yours):
#   - 4 nodes reachable by ssh: ranks .1/.2/.3/.5 (rank0=.1 serves the API on :8000), 200G fabric GID3.
#   - Node .3 holds / NFS-exports the checkpoint at $MODEL_HOST, the other nodes mount it read-only.
#   - Engine image pulled from GHCR; plugin + patches come from this repo's serve/ dir (bind-mounted).
#   - Weights from HuggingFace: drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid (~410 GB).
#
# Usage:  bash oneshot.sh            # full: pull image on all nodes, fetch weights if missing, launch
#         SKIP_WEIGHTS=1 bash oneshot.sh   # weights already on disk
set -u
IMAGE_REMOTE="ghcr.io/drowzeys/vllm-dsv41-overlay5-e47aa:serving-node1"
IMAGE_LOCAL="vllm-dsv41:overlay5-e47aa"          # tag the launcher expects
NODES=(10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5)
MODEL_HF="drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid"
MODEL_NAME="DeepSeek-V4.1-Flash-TR3-Hybrid"
MODEL_HOST="/home/keyspark/models/$MODEL_NAME"      # on .3; exported over NFS to the others
REPO="$(cd "$(dirname "$0")" && pwd)"
log(){ echo "[$(date +%T)] $*"; }

# 1) engine image on every node (pull from GHCR, tag to the local name the launcher uses).
#    The GHCR package is private by default -> authenticate. Set GHCR_USER + GHCR_TOKEN
#    (a token with read:packages), or the script falls back to `gh auth token`.
GHCR_USER="${GHCR_USER:-drowzeys}"
GHCR_TOKEN="${GHCR_TOKEN:-$(gh auth token 2>/dev/null)}"
for n in "${NODES[@]}"; do
  log "image on $n"
  ssh "$n" "docker image inspect $IMAGE_LOCAL >/dev/null 2>&1 && exit 0
    [ -n '$GHCR_TOKEN' ] && echo '$GHCR_TOKEN' | docker login ghcr.io -u '$GHCR_USER' --password-stdin >/dev/null 2>&1
    docker pull $IMAGE_REMOTE && docker tag $IMAGE_REMOTE $IMAGE_LOCAL" \
    || { log "FAILED image on $n (private GHCR pkg? set GHCR_TOKEN, or make the package public in its GitHub Settings)"; exit 1; }
done

# 2) weights on .3 (hf download; shards 1,2,43-48 + small files are release-identical and can be hardlinked)
if [ "${SKIP_WEIGHTS:-0}" != 1 ]; then
  log "fetching weights $MODEL_HF -> .3:$MODEL_HOST (~410 GB, one time)"
  ssh 10.100.10.3 "export HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN=\$(cat ~/.cache/huggingface/token 2>/dev/null); \
    mkdir -p '$MODEL_HOST'; ~/.local/bin/hf download '$MODEL_HF' --local-dir '$MODEL_HOST' --max-workers 16" \
    || { log "weights download FAILED (set HF token or SKIP_WEIGHTS=1 if already present)"; exit 1; }
fi
ssh 10.100.10.3 "test -f '$MODEL_HOST/model-00048-of-00048.safetensors'" || { log "checkpoint incomplete on .3"; exit 1; }

# 3) stage the launcher + plugin + patches from this repo onto every node (the launcher bind-mounts serve/)
for n in "${NODES[@]}"; do
  ssh "$n" "mkdir -p ~/tr3-serve"
  rsync -a --delete "$REPO/serve/" "$n:~/tr3-serve/" >/dev/null 2>&1 || scp -qr "$REPO/serve/"* "$n:~/tr3-serve/"
done

# 4) launch the CURRENT-SERVE config (cluster.py fans out to all ranks)
log "launching TR3-Hybrid (bm8+fast_math) @1M on 4 Sparks"
cd ~/tr3-serve 2>/dev/null || cd "$REPO/serve"
IMAGE="$IMAGE_LOCAL" PATCH_SET=patch-upstream-boot10 TR3=1 CACHE_TAG=tr3 \
  EAGER=0 SPEC=1 TEXT_ONLY=0 EXTRA=0 GMU=0.80 MAXLEN=1048576 SEQS=8 BATCH=8192 \
  TR3_FAST_MATH=1 TR3_DECODE_BLOCK_M=8 TR3_PREFILL_BLOCK_M=64 \
  python3 cluster.py start

# 5) wait for readiness, then smoke test
log "waiting for API (rank0 .1:8000, ~13-18 min cold)"
for i in $(seq 1 180); do
  curl -sf -m 5 http://10.100.10.1:8000/v1/models >/dev/null 2>&1 && { log "SERVING"; break; }
  sleep 10
done
curl -sf -m 5 http://10.100.10.1:8000/v1/models >/dev/null 2>&1 \
  && python3 functest.py http://10.100.10.1:8000 2>&1 | tail -2 \
  || { log "NOT serving after wait; check: python3 cluster.py logs"; exit 1; }
log "Hermes-shaped warmup (kills first-prompt 20-200s TTFT)"
python3 "$REPO/serve/hermes-warmup.py" --api http://10.100.10.1:8000/v1 || python3 hermes-warmup.py --api http://10.100.10.1:8000/v1 || true
log "done. model=deepseek-v4.1-flash endpoint=http://10.100.10.1:8000"
log "Hermes: see HERMES.md — max_tokens 12288, reasoning_effort: false, /new after point"
