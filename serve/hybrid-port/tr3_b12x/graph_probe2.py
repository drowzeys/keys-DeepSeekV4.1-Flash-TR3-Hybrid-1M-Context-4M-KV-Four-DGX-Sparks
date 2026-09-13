"""Does graph REPLAY write into memory freed after capture? Capture per tier, fill freed pool with canaries, replay, verify."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Tr3LayerExperts, TILE, K_WORDS
L, R = 3, 0; model = Path("/model"); H, I, E, TOPK = 5120, 2304, 384, 6; dev = torch.device("cuda:0")
from safetensors import safe_open
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
def load(name):
    with safe_open(str(model / index[name]), framework="pt", device="cpu") as f: return f.get_tensor(name)
local = list(range(R * 96, (R + 1) * 96)); keep = [e for e in local if f"layers.{L}.ffn.experts.{e}.w1.weight" in index]; tail = [e for e in local if e not in keep]
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
lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0, dtype=torch.bfloat16, device=dev, decode_tokens=8, prefill_tokens=4096, shared_scratch={})
lay.set_local_ids(local)
lay.build_tail(global_ids=tail, w13_trellis=w13t, w2_trellis=w2t, gate_suh=gsuh, up_suh=usuh, intermediate_rotations=ir, down_svh=dsvh, mcg=mcg)
lay.build_keep(global_ids=keep, w13_fp4=w13k, w13_scale=w13s, w2_fp4=w2k, w2_scale=w2s)
# weight fingerprints (the offline analogue of the in-model check)
fps = {n: float(t.view(torch.uint8).float().sum()) for n, t in dict(w13t=w13t, w2t=w2t, gsuh=gsuh, ir=ir, w13k=w13k, w2k=w2k).items()}
def canaries():
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    cs = [torch.full((n,), 0x5A, dtype=torch.uint8, device=dev) for n in (4096, 65536, 1 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30)]
    torch.cuda.synchronize(); return cs
def check(cs, tag):
    torch.cuda.synchronize(); bad = []
    for c in cs:
        d = (c != 0x5A).nonzero()
        if d.numel(): bad.append(f"canary {c.numel()//1024}KB: {int(d.numel())} bytes changed, first at {int(d[0])}")
    ch = [n for n, t in dict(w13t=w13t, w2t=w2t, gsuh=gsuh, ir=ir, w13k=w13k, w2k=w2k).items() if float(t.view(torch.uint8).float().sum()) != fps[n]]
    print(json.dumps(dict(tag=tag, canaries_hit=bad, weights_changed=ch)), flush=True)
for tier_name in ("tail", "keep"):
    tier = getattr(lay, tier_name)
    for T, kind in ((4, "decode"), (128, "prefill")):
        x = torch.randn(T, H, device=dev).half(); ids = torch.randint(0, E, (T, TOPK), device=dev).long(); ids[:, 0] = torch.tensor(local, device=dev)[torch.randint(0, 96, (T,), device=dev)]; w = torch.rand(T, TOPK, device=dev)
        tier.run(x, w, ids, kind); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = tier.run(x, w, ids, kind)
        cs = canaries()
        for _ in range(5): g.replay()
        check(cs, f"{tier_name} T={T} {kind} replay x5")
        del cs
        # eager control with canaries present
        cs = canaries(); tier.run(x, w, ids, kind); check(cs, f"{tier_name} T={T} {kind} eager"); del cs, g
