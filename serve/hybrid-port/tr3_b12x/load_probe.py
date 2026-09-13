"""Replicate the vLLM load flow for one rank: pre-allocate all 40 layers' params (torch.empty, cuda),
iterate EVERY checkpoint tensor via safetensors get_tensor, copy local experts in. Log MemAvailable.
usage: load_probe.py RANK [zeros|empty]"""
import json, os, sys, time, re
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import TILE, K_WORDS
R = int(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else "empty"
model = Path("/model"); H, I, E = 5120, 2304, 384; dev = torch.device("cuda:0")
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
def avail():
    with open("/proc/meminfo") as f: mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return mi["MemAvailable"] / 2**20
local = list(range(R * 96, (R + 1) * 96)); alloc = torch.zeros if MODE == "zeros" else torch.empty
P = {}
for L in range(40):
    keep = [e for e in local if f"layers.{L}.ffn.experts.{e}.w1.weight" in index]; tail = [e for e in local if e not in keep]
    Et, Ek = len(tail), len(keep)
    P[L] = dict(tail={g: i for i, g in enumerate(tail)}, keep={g: i for i, g in enumerate(keep)},
        w13t=alloc(2, Et, H // TILE, I // TILE, K_WORDS, dtype=torch.int16, device=dev), w2t=alloc(Et, I // TILE, H // TILE, K_WORDS, dtype=torch.int16, device=dev),
        gsuh=alloc(Et, H, dtype=torch.float16, device=dev), usuh=alloc(Et, H, dtype=torch.float16, device=dev), dsvh=alloc(Et, H, dtype=torch.float16, device=dev),
        ir=alloc(Et, 3 * I, dtype=torch.float16, device=dev), mcg=torch.zeros(Et, 3, dtype=torch.int32, device=dev),
        w13k=alloc(Ek, 2 * I, H // 2, dtype=torch.uint8, device=dev), w13s=alloc(Ek, 2 * I, H // 32, dtype=torch.uint8, device=dev),
        w2k=alloc(Ek, H, I // 2, dtype=torch.uint8, device=dev), w2s=alloc(Ek, H, I // 32, dtype=torch.uint8, device=dev))
torch.cuda.synchronize(); print(f"after create ({MODE}): gpu_alloc {torch.cuda.memory_allocated()/2**30:.1f}G avail {avail():.1f}G", flush=True)
rx = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mcg|weight|scale)$")
shards = sorted(set(index.values())); n = 0; nbytes = 0; t0 = time.time(); copied = 0
for sh in shards:
    with safe_open(str(model / sh), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name); n += 1; nbytes += t.numel() * t.element_size()
            m = rx.match(name)
            if m:
                L, g, w, k = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4); p = P[L]
                if g in p["tail"] and k in ("trellis", "suh", "svh", "mcg"):
                    s = p["tail"][g]; copied += 1
                    if k == "trellis": (p["w13t"][0 if w == "w1" else 1, s] if w != "w2" else p["w2t"][s]).copy_(t)
                    elif k == "suh": ({"w1": p["gsuh"], "w3": p["usuh"]}[w][s] if w != "w2" else p["ir"][s, 2 * I:]).copy_(t)
                    elif k == "svh": (p["ir"][s, :I] if w == "w1" else p["ir"][s, I:2 * I] if w == "w3" else p["dsvh"][s]).copy_(t)
                    else: p["mcg"][s, {"w1": 0, "w3": 1, "w2": 2}[w]] = int(t.reshape(-1)[0])
                elif g in p["keep"] and k in ("weight", "scale"):
                    s = p["keep"][g]; copied += 1; u = t.view(torch.uint8)
                    if k == "weight": (p["w13k"][s, :I] if w == "w1" else p["w13k"][s, I:] if w == "w3" else p["w2k"][s]).copy_(u)
                    else: (p["w13s"][s, :I] if w == "w1" else p["w13s"][s, I:] if w == "w3" else p["w2s"][s]).copy_(u)
            if n % 10000 == 0:
                torch.cuda.synchronize()
                print(f"{n} tensors {nbytes/2**30:.0f}G read, {copied} copied, {time.time()-t0:.0f}s: gpu_alloc {torch.cuda.memory_allocated()/2**30:.1f}G reserved {torch.cuda.memory_reserved()/2**30:.1f}G avail {avail():.1f}G", flush=True)
                if avail() < 25: print("ABORT: avail < 25G", flush=True); sys.exit(3)
torch.cuda.synchronize(); print(f"DONE {n} tensors, {copied} copied, {time.time()-t0:.0f}s: gpu_alloc {torch.cuda.memory_allocated()/2**30:.1f}G avail {avail():.1f}G", flush=True)
