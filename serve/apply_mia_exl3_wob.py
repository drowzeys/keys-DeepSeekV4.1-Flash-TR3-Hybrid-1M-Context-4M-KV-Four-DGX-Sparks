#!/usr/bin/env python3
"""Splice Mia EXL3 K=5 mul1 L10-35 wo_b sidecar into a NEW copy of the 2.9bpw pack.

Never writes stock. Hardlinks unchanged shards.
Sidecar keys: layers.{10-35}.attn.wo_b.{trellis,suh,svh,mul1}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYERS = range(10, 36)
SUFS = ("trellis", "suh", "svh", "mul1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw dir")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--wo-b", type=Path, required=True, help="mia_exl3_wo_b_l10_35.safetensors")
    args = ap.parse_args()
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("refusing to write through --src")

    graft = {}
    with safe_open(str(args.wo_b), framework="pt") as f:
        for k in f.keys():
            graft[k] = f.get_tensor(k)
    wanted = [f"layers.{i}.attn.wo_b.{s}" for i in LAYERS for s in SUFS]
    missing = [k for k in wanted if k not in graft]
    if missing:
        raise SystemExit(f"sidecar missing {missing[:6]}")

    wm = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    sample = "layers.20.attn.wo_b.trellis"
    if sample not in wm:
        raise SystemExit(f"{args.src} is not the Mia EXL3 2.9bpw pack (no {sample})")

    args.dst.mkdir(parents=True, exist_ok=True)
    by_shard = defaultdict(list)
    for name, shard in wm.items():
        by_shard[shard].append(name)

    for p in args.src.iterdir():
        if p.name.startswith("model-") and p.name.endswith(".safetensors"):
            continue
        if p.name in ("model.safetensors.index.json", ".cache"):
            continue
        dest = args.dst / p.name
        if p.is_dir():
            if not dest.exists():
                shutil.copytree(p, dest, dirs_exist_ok=True)
        elif not dest.exists():
            shutil.copy2(p, dest)

    edit = set(wanted)
    n_edit = 0
    for shard_name, keys in sorted(by_shard.items()):
        src_path = args.src / shard_name
        dst_path = args.dst / shard_name
        needs = [k for k in keys if k in edit]
        if not needs:
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
            continue
        print(f"edit {shard_name} {needs}", flush=True)
        tensors = {}
        with safe_open(str(src_path), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k in needs:
            tensors[k] = graft[k].contiguous()
            n_edit += 1
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        save_file({k: t.contiguous() if torch.is_tensor(t) else t for k, t in tensors.items()}, str(tmp))
        os.chmod(tmp, 0o644)
        tmp.replace(dst_path)

    shutil.copy2(args.src / "model.safetensors.index.json", args.dst / "model.safetensors.index.json")
    (args.dst / "ABLIT_META.json").write_text(json.dumps({
        "method": "graft-keys-l10-35-wo_b-exl3-mul1-k5",
        "n_edited": n_edit,
        "source_stock": str(args.src),
        "sidecar": str(args.wo_b),
        "serve": "point MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks MODEL_HOST at --dst",
    }, indent=2))
    print("DONE", args.dst, "edited", n_edit)


if __name__ == "__main__":
    main()
