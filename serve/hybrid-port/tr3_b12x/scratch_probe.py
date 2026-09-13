"""Does leftover scratch content or input mutation explain in-model NaN? One real layer, rank 0."""
import json, os, sys, time
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
shared = {}
lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0, dtype=torch.bfloat16, device=dev, decode_tokens=8, prefill_tokens=4096, shared_scratch=shared)
lay.set_local_ids(local)
lay.build_tail(global_ids=tail, w13_trellis=w13t, w2_trellis=w2t, gate_suh=gsuh, up_suh=usuh, intermediate_rotations=ir, down_svh=dsvh, mcg=mcg)
lay.build_keep(global_ids=keep, w13_fp4=w13k, w13_scale=w13s, w2_fp4=w2k, w2_scale=w2s)
torch.manual_seed(0)
def run(T, tag, poison=None):
    x = torch.randn(T, H, device=dev).bfloat16(); ids = torch.randint(0, E, (T, TOPK), device=dev); w = torch.rand(T, TOPK, device=dev)
    ids[0, 0] = local[0]  # one guaranteed local hit; other rows may have none
    x16 = x.to(torch.float16); x_before = x16.clone()
    if poison is not None:
        for k, buf in shared.items():
            buf.view(torch.uint8).fill_(poison)  # 0xFF bytes = NaN in fp16/fp32, -1 in ints
    out = lay.forward(x, w, ids); torch.cuda.synchronize()
    nonfinite = int((~torch.isfinite(out)).sum()); rows_nan = int((~torch.isfinite(out)).any(dim=1).sum())
    print(json.dumps(dict(tag=tag, T=T, kind=lay.kind_for(T), poison=poison, nonfinite=nonfinite, rows_nonfinite=rows_nan, out_absmax=float(out.float().abs().max()) if nonfinite == 0 else None)), flush=True)
    return out
run(5, "fresh scratch"); run(5, "second call (leftover scratch)"); run(6, "third")
run(5, "scratch poisoned 0xFF", poison=0xFF); run(5, "after poison, leftover"); run(5, "scratch poisoned 0x7C", poison=0x7C)
run(700, "prefill fresh-ish"); run(700, "prefill poisoned 0xFF", poison=0xFF); run(5, "decode after prefill")
# input mutation check on the tail tier directly
x = torch.randn(5, H, device=dev).half(); xc = x.clone(); ids = torch.randint(0, E, (5, TOPK), device=dev).long(); ids[:, 0] = local[0]; w = torch.rand(5, TOPK, device=dev)
lay.tail.run(x, w, ids, "decode"); torch.cuda.synchronize(); print("tail kernel modified its input a:", not torch.equal(x, xc), flush=True)
lay.keep.run(x, w, ids, "decode"); torch.cuda.synchronize(); print("keep kernel modified its input a:", not torch.equal(x, xc), flush=True)
