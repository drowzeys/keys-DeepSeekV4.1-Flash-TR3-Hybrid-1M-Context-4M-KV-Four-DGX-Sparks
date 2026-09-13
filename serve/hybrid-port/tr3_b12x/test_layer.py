"""Offline check of Tr3LayerExperts on one real TR3 layer, one rank's 96 experts.
Reference: ExLlamaV3 decode (trellis) / MXFP4 dequant (keep) in fp32, only local experts.
usage: python3 test_layer.py LAYER RANK [tokens...]"""
import json, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Tr3LayerExperts, TILE, K_WORDS

L, R = int(sys.argv[1]), int(sys.argv[2]); TOK = [int(t) for t in sys.argv[3:]] or [5, 6, 48, 128, 700]
model = Path("/model"); H, I, E, TOPK, WORLD = 5120, 2304, 384, 6, 4
dev = torch.device("cuda:0"); dtype = torch.bfloat16
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
def load(name):
    with safe_open(str(model / index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)
local = list(range(R * (E // WORLD), (R + 1) * (E // WORLD)))
keep = [e for e in local if f"layers.{L}.ffn.experts.{e}.w1.weight" in index]
tail = [e for e in local if e not in keep]
print(f"layer {L} rank {R}: {len(tail)} tail + {len(keep)} keep experts", flush=True)
t0 = time.time()
Et, Ek = len(tail), len(keep)
w13t = torch.empty(2, Et, H // TILE, I // TILE, K_WORDS, dtype=torch.int16); w2t = torch.empty(Et, I // TILE, H // TILE, K_WORDS, dtype=torch.int16)
gsuh = torch.empty(Et, H, dtype=torch.float16); usuh = torch.empty_like(gsuh); dsvh = torch.empty_like(gsuh)
ir = torch.empty(Et, 3 * I, dtype=torch.float16); mcg = torch.empty(Et, 3, dtype=torch.int32)
for s, e in enumerate(tail):
    p = f"layers.{L}.ffn.experts.{e}"
    w13t[0, s] = load(f"{p}.w1.trellis"); w13t[1, s] = load(f"{p}.w3.trellis"); w2t[s] = load(f"{p}.w2.trellis")
    gsuh[s] = load(f"{p}.w1.suh"); usuh[s] = load(f"{p}.w3.suh"); dsvh[s] = load(f"{p}.w2.svh")
    ir[s, :I] = load(f"{p}.w1.svh"); ir[s, I:2 * I] = load(f"{p}.w3.svh"); ir[s, 2 * I:] = load(f"{p}.w2.suh")
    mcg[s] = torch.stack([load(f"{p}.w{k}.mcg").reshape(()) for k in (1, 3, 2)])
w13k = torch.empty(Ek, 2 * I, H // 2, dtype=torch.uint8); w13s = torch.empty(Ek, 2 * I, H // 32, dtype=torch.uint8)
w2k = torch.empty(Ek, H, I // 2, dtype=torch.uint8); w2s = torch.empty(Ek, H, I // 32, dtype=torch.uint8)
for s, e in enumerate(keep):
    p = f"layers.{L}.ffn.experts.{e}"
    w13k[s, :I] = load(f"{p}.w1.weight").view(torch.uint8); w13k[s, I:] = load(f"{p}.w3.weight").view(torch.uint8)
    w13s[s, :I] = load(f"{p}.w1.scale").view(torch.uint8); w13s[s, I:] = load(f"{p}.w3.scale").view(torch.uint8)
    w2k[s] = load(f"{p}.w2.weight").view(torch.uint8); w2s[s] = load(f"{p}.w2.scale").view(torch.uint8)
print(f"loaded in {time.time()-t0:.0f}s; tail {w13t.numel()*2/2**30 + w2t.numel()*2/2**30:.2f} GiB keep {(w13k.numel()+w2k.numel())/2**30:.2f} GiB", flush=True)
lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0,
                      dtype=dtype, device=dev, decode_tokens=48, prefill_tokens=4096)
lay.set_local_ids(local)
t0 = time.time()
lay.build_tail(global_ids=tail, w13_trellis=w13t.to(dev), w2_trellis=w2t.to(dev), gate_suh=gsuh.to(dev), up_suh=usuh.to(dev),
               intermediate_rotations=ir.to(dev), down_svh=dsvh.to(dev), mcg=mcg.to(dev))
print(f"tail built in {time.time()-t0:.0f}s", flush=True); t0 = time.time()
lay.build_keep(global_ids=keep, w13_fp4=w13k.to(dev), w13_scale=w13s.to(dev), w2_fp4=w2k.to(dev), w2_scale=w2s.to(dev))
print(f"keep built in {time.time()-t0:.0f}s; GPU mem {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

# ---- references ---------------------------------------------------------
import exllamav3_ext as ext
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def deq_fp4(packed, scale):  # [N,K/2] u8, [N,K/32] e8m0 -> [N,K] f32
    N = packed.shape[0]; idx = torch.stack(((packed & 0xF).long(), (packed >> 4).long()), -1).view(N, -1)
    w = LUT[idx]; sc = scale.view(torch.float8_e8m0fnu).float()
    return (w.view(N, -1, 32) * sc.unsqueeze(-1)).view(N, -1)
def deq_trellis(tr, suh, svh, k, n):
    out = torch.empty((k, n), dtype=torch.float16, device=dev)
    ext.reconstruct_had_slice(out, tr.contiguous(), suh.contiguous(), svh.contiguous(), 3, True, False, 0); return out.float()
ref_cache = {}
def expert_mats(g):  # returns (W1 [H,I], W3 [H,I], W2 [I,H]) fp32 as x @ W
    if g in ref_cache: return ref_cache[g]
    if g in tail:
        s = tail.index(g)
        W1 = deq_trellis(w13t[0, s].to(dev), gsuh[s].to(dev), ir[s, :I].to(dev), H, I)
        W3 = deq_trellis(w13t[1, s].to(dev), usuh[s].to(dev), ir[s, I:2*I].to(dev), H, I)
        W2 = deq_trellis(w2t[s].to(dev), ir[s, 2*I:].to(dev), dsvh[s].to(dev), I, H)
    else:
        s = keep.index(g)
        W1 = deq_fp4(w13k[s, :I].to(dev), w13s[s, :I].to(dev)).t(); W3 = deq_fp4(w13k[s, I:].to(dev), w13s[s, I:].to(dev)).t()
        W2 = deq_fp4(w2k[s].to(dev), w2s[s].to(dev)).t()
    ref_cache[g] = (W1, W3, W2); return ref_cache[g]
# cross-check the fp4 convention with vLLM's own dequant op on one keep expert
try:
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _dequant_mxfp4
    s = 0; mine = deq_fp4(w2k[s].to(dev), w2s[s].to(dev))
    theirs = _dequant_mxfp4(w2k[s].to(dev), w2s[s].to(dev).view(torch.float8_e8m0fnu), torch.bfloat16).float()
    print("fp4 convention vs vLLM _dequant_mxfp4: max abs diff", (mine - theirs).abs().max().item(), flush=True)
except Exception as e:
    print("vLLM dequant cross-check skipped:", str(e)[:120], flush=True)

import os
MODE = os.environ.get("MODE", "MIX")  # MIX | TAIL | KEEP
pool = {"MIX": local, "TAIL": tail, "KEEP": keep}[MODE]
torch.manual_seed(0); bad = 0
for T in TOK:
    x = (torch.randn(T, H, device=dev) * 1.0).to(dtype)
    ids = torch.randint(0, E, (T, TOPK), device=dev, dtype=torch.int64)
    if MODE != "MIX":  # every slot from the chosen tier (or non-local)
        pick = torch.tensor(pool, device=dev)[torch.randint(0, len(pool), (T, TOPK), device=dev)]
        nonlocal_ = torch.tensor([e for e in range(E) if e not in local], device=dev)[torch.randint(0, E - len(local), (T, TOPK), device=dev)]
        ids = torch.where(torch.rand(T, TOPK, device=dev) < 0.5, pick, nonlocal_)
    ids[:, 0] = torch.tensor([pool[i % len(pool)] for i in range(T)], device=dev)  # ensure hits
    if os.environ.get("NOLOCAL_ROWS") == "1":  # every other row gets NO local expert at all (serving-realistic)
        nonloc = torch.tensor([e for e in range(E) if e not in local], device=dev)
        for t in range(0, T, 2):
            ids[t] = nonloc[torch.randint(0, len(nonloc), (TOPK,), device=dev)]
    w = torch.softmax(torch.randn(T, TOPK, device=dev), -1) * 1.5
    torch.cuda.synchronize(); t0 = time.time(); out = lay.forward(x, w, ids); torch.cuda.synchronize(); dt = time.time() - t0
    ref = torch.zeros(T, H, device=dev)
    xf = x.float()
    for t in range(T):
        for j in range(TOPK):
            g = int(ids[t, j])
            if g not in local: continue
            W1, W3, W2 = expert_mats(g)
            gate = (xf[t] @ W1).clamp(max=10.0); up = (xf[t] @ W3).clamp(-10.0, 10.0)
            ref[t] += w[t, j] * ((F.silu(gate) * up) @ W2)
    err = ((out.float() - ref).norm() / (ref.norm() + 1e-9)).item(); cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), 0).item()
    nonfinite = int((~torch.isfinite(out)).sum()); ok = nonfinite == 0 and err < 0.03 and cos > 0.999; bad += 0 if ok else 1
    per_tok = ((out.float() - ref).norm(dim=1) / (ref.norm(dim=1) + 1e-6))
    has_keep = torch.tensor([any(int(g) in keep for g in ids[t]) for t in range(T)], device=dev)
    print(f"   tok err>5%: {int((per_tok > 0.05).sum())}/{T}; mean err tokens w/ keep hit {per_tok[has_keep].mean().item() if has_keep.any() else float('nan'):.4f}, w/o {per_tok[~has_keep].mean().item() if (~has_keep).any() else float('nan'):.4f}", flush=True)
    # graph replay
    g = torch.cuda.CUDAGraph(); lay.forward(x, w, ids); torch.cuda.synchronize()
    with torch.cuda.graph(g): cap = lay.forward(x, w, ids)
    g.replay(); torch.cuda.synchronize(); gdiff = (cap.float() - out.float()).abs().max().item()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t0 = time.time(); [g.replay() for _ in range(10)]; torch.cuda.synchronize(); gt = (time.time() - t0) / 10
    print(json.dumps(dict(T=T, kind=lay.kind_for(T), rel_err=round(err, 5), cos=round(cos, 7), nonfinite=nonfinite, graph_maxdiff=gdiff,
                          eager_ms=round(dt * 1e3, 2), graph_ms=round(gt * 1e3, 3), ok=ok)), flush=True)
print("RESULT", "PASS" if bad == 0 else f"FAIL {bad}")
