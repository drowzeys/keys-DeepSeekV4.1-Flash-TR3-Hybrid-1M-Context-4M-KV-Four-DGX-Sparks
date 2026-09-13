#!/usr/bin/env bash
# compute-sanitizer memcheck of the single-layer test, sanitizer mounted from the host (CUDA 13.0). Run on .3 with a free GPU.
set -euo pipefail
HERE=/home/keyspark/dsv41-recovery-20260912
docker run --rm --gpus all --memory 60g --memory-swap 60g --oom-score-adj 900 --entrypoint bash \
  -v "$HERE/hybrid-port:/work:ro" -v /home/keyspark/models/DeepSeek-V4.1-Flash-TR3-Hybrid:/model:ro \
  -v "$HERE/hybrid-port/b12x:/brandon/b12x:ro" -v /home/keyspark/models/dsv41-exl3-tp4/exl3-graft:/ext:ro \
  -v /home/keyspark/dsv41-native-cache:/cache -v /usr/local/cuda-13.0/compute-sanitizer:/sanitizer:ro \
  -e PYTHONPATH=/brandon:/ext -e TORCH_CUDA_ARCH_LIST=12.1a -e TRITON_CACHE_DIR=/cache/triton-b12x -e NOLOCAL_ROWS=1 \
  vllm-dsv41:overlay5 -c "cd /work/tr3_b12x && /sanitizer/compute-sanitizer --tool memcheck --leak-check no --report-api-errors no --print-limit 60 --error-exitcode 9 python3 test_layer.py $*"
