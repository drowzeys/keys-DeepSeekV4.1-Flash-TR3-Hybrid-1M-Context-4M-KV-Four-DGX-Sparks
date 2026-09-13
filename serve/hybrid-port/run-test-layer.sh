#!/usr/bin/env bash
# usage: run-test-layer.sh LAYER RANK [tokens...]   (run on 10.100.10.3)
set -euo pipefail
HERE=/home/keyspark/dsv41-recovery-20260912
docker run --rm --gpus all --memory 60g --memory-swap 60g --oom-score-adj 900 --entrypoint bash \
  -v "$HERE/hybrid-port:/work:ro" -v /home/keyspark/models/DeepSeek-V4.1-Flash-TR3-Hybrid:/model:ro \
  -v "$HERE/hybrid-port/b12x:/brandon/b12x:ro" -v /home/keyspark/models/dsv41-exl3-tp4/exl3-graft:/ext:ro \
  -v /home/keyspark/dsv41-native-cache:/cache \
  -e PYTHONPATH=/brandon:/ext -e TORCH_CUDA_ARCH_LIST=12.1a -e TRITON_CACHE_DIR=/cache/triton-b12x \
  vllm-dsv41:overlay5 -c "cd /work/tr3_b12x && python3 test_layer.py $*"
