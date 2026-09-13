"""Build N layers sequentially like the plugin does; print host RSS / free and GPU allocated per layer."""
import json, os, sys, time, resource
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Tr3LayerExperts, TILE, K_WORDS
N, R = int(sys.argv[1]), int(sys.argv[2]); model = Path("/model"); H, I, E, TOPK = 5120, 2304, 384, 6
dev = torch.device("cuda:0")
from safetensors import safe_open
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
def load(name):
    with safe_open(str(model / index[name]), framework="pt", device="cpu") as f: return f.get_tensor(name)
def host():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    with open("/proc/meminfo") as f: mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return f"maxrss {rss:.1f}G avail {mi['MemAvailable']/2**20:.1f}G"
shared = {}; layers = []
local = list(range(R * 96, (R + 1) * 96))
for L in range(N):
    t0 = time.time()
    keep = [e for e in local if f"layers.{L}.ffn.experts.{e}.w1.weight" in index]; tail = [e for e in local if e not in keep]
    Et, Ek = len(tail), len(keep)
    w13t = torch.zeros(2, Et, H // TILE, I // TILE, K_WORDS, dtype=torch.int16, device=dev); w2t = torch.zeros(Et, I // TILE, H // TILE, K_WORDS, dtype=torch.int16, device=dev)
    gsuh = torch.zeros(Et, H, dtype=torch.float16, device=dev); usuh = torch.zeros_like(gsuh); dsvh = torch.zeros_like(gsuh); ir = torch.zeros(Et, 3 * I, dtype=torch.float16, device=dev); mcg = torch.zeros(Et, 3, dtype=torch.int32, device=dev)
    w13k = torch.zeros(Ek, 2 * I, H // 2, dtype=torch.uint8, device=dev); w13s = torch.zeros(Ek, 2 * I, H // 32, dtype=torch.uint8, device=dev); w2k = torch.zeros(Ek, H, I // 2, dtype=torch.uint8, device=dev); w2s = torch.zeros(Ek, H, I // 32, dtype=torch.uint8, device=dev)
    for s, e in enumerate(tail):
        p = f"layers.{L}.ffn.experts.{e}"; w13t[0, s] = load(f"{p}.w1.trellis").to(dev); w13t[1, s] = load(f"{p}.w3.trellis").to(dev); w2t[s] = load(f"{p}.w2.trellis").to(dev)
        gsuh[s] = load(f"{p}.w1.suh").to(dev); usuh[s] = load(f"{p}.w3.suh").to(dev); dsvh[s] = load(f"{p}.w2.svh").to(dev); ir[s, :I] = load(f"{p}.w1.svh").to(dev); ir[s, I:2*I] = load(f"{p}.w3.svh").to(dev); ir[s, 2*I:] = load(f"{p}.w2.suh").to(dev)
        mcg[s] = torch.stack([load(f"{p}.w{k}.mcg").reshape(()) for k in (1, 3, 2)]).to(dev)
    for s, e in enumerate(keep):
        p = f"layers.{L}.ffn.experts.{e}"; w13k[s, :I] = load(f"{p}.w1.weight").view(torch.uint8).to(dev); w13k[s, I:] = load(f"{p}.w3.weight").view(torch.uint8).to(dev)
        w13s[s, :I] = load(f"{p}.w1.scale").view(torch.uint8).to(dev); w13s[s, I:] = load(f"{p}.w3.scale").view(torch.uint8).to(dev); w2k[s] = load(f"{p}.w2.weight").view(torch.uint8).to(dev); w2s[s] = load(f"{p}.w2.scale").view(torch.uint8).to(dev)
    torch.cuda.synchronize(); a0 = torch.cuda.memory_allocated() / 2**30; h0 = host()
    lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0, dtype=torch.bfloat16, device=dev, decode_tokens=24, prefill_tokens=4096, shared_scratch=shared)
    lay.build_tail(global_ids=tail, w13_trellis=w13t, w2_trellis=w2t, gate_suh=gsuh, up_suh=usuh, intermediate_rotations=ir, down_svh=dsvh, mcg=mcg)
    lay.build_keep(global_ids=keep, w13_fp4=w13k, w13_scale=w13s, w2_fp4=w2k, w2_scale=w2s)
    del w13k, w13s, w2k, w2s
    layers.append(lay); torch.cuda.synchronize()
    print(json.dumps(dict(layer=L, tail=Et, keep=Ek, gpu_after_params=round(a0, 2), gpu_after_build=round(torch.cuda.memory_allocated() / 2**30, 2),
                          gpu_reserved=round(torch.cuda.memory_reserved() / 2**30, 2), host_before=h0, host_after=host(), secs=round(time.time() - t0, 1))), flush=True)
x = torch.randn(6, H, device=dev).bfloat16(); ids = torch.randint(0, E, (6, TOPK), device=dev); w = torch.rand(6, TOPK, device=dev)
for lay in layers: lay.forward(x, w, ids)
torch.cuda.synchronize(); print("all layers forward ok;", host(), flush=True)
