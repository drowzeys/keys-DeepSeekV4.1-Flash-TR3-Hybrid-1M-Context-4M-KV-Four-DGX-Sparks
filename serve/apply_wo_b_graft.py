#!/usr/bin/env python3
"""Overlay Keys L10-35 attn.wo_b onto native / EXL3-3.5bpw / stock TR3.

Attention is official FP8 on all three packs (byte-identical wo_b before ablit).
This copies ONLY layers.10-35 attn.wo_b.{weight,scale} from a small sidecar
into a NEW dest. Everything else is hardlinked from --src (never write stock).

Sidecar: wo_b_l10_35.safetensors next to this script, or --wo-b PATH.
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


def _load_sidecar(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        cands = list(path.glob("wo_b_l10_35.safetensors")) + list(path.glob("*.safetensors"))
        if not cands:
            raise SystemExit(f"no safetensors in {path}")
        path = cands[0]
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="stock native, EXL3 3.5bpw, or TR3-Hybrid dir")
    ap.add_argument("--dst", type=Path, required=True, help="NEW dir (created). never --src")
    ap.add_argument("--wo-b", type=Path, default=None, help="sidecar safetensors or directory")
    args = ap.parse_args()
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("refusing to write through --src")

    here = Path(__file__).resolve().parent
    wob_path = args.wo_b or here / "wo_b_l10_35.safetensors"
    graft = _load_sidecar(wob_path)
    wanted = []
    for lid in LAYERS:
        for suf in (".weight", ".scale"):
            n = f"layers.{lid}.attn.wo_b{suf}"
            if n not in graft:
                raise SystemExit(f"sidecar missing {n}")
            wanted.append(n)
    print(f"sidecar {wob_path} n={len(wanted)}")

    idx = json.loads((args.src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    sample = "layers.20.attn.wo_b.weight"
    if sample not in wm:
        raise SystemExit(f"{args.src} has no {sample} — not a V4.1 Flash checkpoint")

    with safe_open(str(args.src / wm[sample]), framework="pt") as f:
        stock = f.get_tensor(sample)
    g = graft[sample]
    print(f"sanity {sample} stock {tuple(stock.shape)} {stock.dtype} graft {tuple(g.shape)} {g.dtype}")
    if tuple(stock.shape) != tuple(g.shape) or stock.dtype != g.dtype:
        raise SystemExit("shape/dtype mismatch — refuse to splice")

    args.dst.mkdir(parents=True, exist_ok=True)
    by_shard = defaultdict(list)
    for name, shard in wm.items():
        by_shard[shard].append(name)

    edit = set(wanted)
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
    meta = {
        "method": "graft-keys-l10-35-wo_b",
        "source_stock": str(args.src),
        "sidecar": str(wob_path),
        "n_edited": n_edit,
        "layers": "10-35",
        "note": "attn.wo_b only. Experts/Engram/MTP/L0-9/L36-39 remain the --src pack.",
    }
    (args.dst / "ABLIT_META.json").write_text(json.dumps(meta, indent=2))
    print("DONE", args.dst, "edited", n_edit)


if __name__ == "__main__":
    main()
