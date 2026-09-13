"""Capture a CUDA graph with routing A, replay with routing B (same shapes). Compare to eager(B) per tier."""
import json, sys
from pathlib import Path
import torch, torch.nn.functional as F
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
torch.manual_seed(0)
def rand_inputs(T, seed, dummy=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, H, device=dev, generator=g).half()
    if dummy:  # capture-time style: all rows route to the same few experts (like vLLM's dummy run)
        ids = torch.arange(TOPK, device=dev).repeat(T, 1).long()
    else:
        ids = torch.randint(0, E, (T, TOPK), device=dev, generator=g).long(); ids[:, 0] = torch.tensor(local, device=dev)[torch.randint(0, 96, (T,), device=dev, generator=g)]
    w = torch.rand(T, TOPK, device=dev, generator=g)
    return x, ids, w
for tier_name in ("tail", "keep"):
    tier = getattr(lay, tier_name)
    for T, kind in ((4, "decode"), (6, "decode"), (128, "prefill")):
        # static buffers the graph will read
        xs, ids_s, ws = rand_inputs(T, 1, dummy=True)
        tier.run(xs, ws, ids_s, kind); torch.cuda.synchronize()          # warm
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap = tier.run(xs, ws, ids_s, kind)
        # replay with DIFFERENT routing/inputs copied into the same buffers
        xb, ids_b, wb = rand_inputs(T, 7)
        xs.copy_(xb); ids_s.copy_(ids_b); ws.copy_(wb)
        g.replay(); torch.cuda.synchronize(); rep = cap.clone()
        ref = tier.run(xs, ws, ids_s, kind).clone(); torch.cuda.synchronize()   # eager with B
        nonfinite = int((~torch.isfinite(rep)).sum()); diff = (rep.float() - ref.float()).abs().max().item() if nonfinite == 0 else float("nan")
        cos = F.cosine_similarity(rep.float().flatten(), ref.float().flatten(), 0).item() if nonfinite == 0 else float("nan")
        print(json.dumps(dict(tier=tier_name, T=T, kind=kind, replayB_nonfinite=nonfinite, maxdiff_vs_eagerB=diff, cos=round(cos, 6))), flush=True)
