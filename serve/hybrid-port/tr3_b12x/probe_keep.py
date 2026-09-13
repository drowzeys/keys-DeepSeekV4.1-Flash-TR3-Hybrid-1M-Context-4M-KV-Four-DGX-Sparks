"""Synthetic probe of the B12X fp4_e8m0_k32 keep path: known float weights packed with
each (nibble order, gate/up row order) hypothesis; report which matches an fp32 reference."""
import itertools, json, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Tr3LayerExperts
dev = torch.device("cuda:0"); H, I, E, TOPK = 5120, 2304, 384, 6
torch.manual_seed(1)
VALS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def rand_codes(n, k): return torch.randint(0, 16, (n, k), device=dev, dtype=torch.uint8)
def rand_scales(n, k): return torch.randint(114, 120, (n, k // 32), device=dev, dtype=torch.uint8)   # e8m0 2^-7..2^4
def deq(codes, sc):  # [N,K] codes, [N,K/32] e8m0 -> f32 [N,K]
    N, K = codes.shape; w = VALS[codes.long()]; s = sc.view(torch.float8_e8m0fnu).float()
    return (w.view(N, K // 32, 32) * s.unsqueeze(-1)).view(N, K)
def pack(codes, low_first):  # [N,K] -> [N,K/2] u8
    a, b = codes[:, 0::2], codes[:, 1::2]
    return (a | (b << 4)) if low_first else (b | (a << 4))
n_local = 2; gids = [7, 300]
c1 = [rand_codes(I, H) for _ in gids]; c3 = [rand_codes(I, H) for _ in gids]; c2 = [rand_codes(H, I) for _ in gids]
s1 = [rand_scales(I, H) for _ in gids]; s3 = [rand_scales(I, H) for _ in gids]; s2 = [rand_scales(H, I) for _ in gids]
T = 16
x = torch.randn(T, H, device=dev).half()
ids = torch.full((T, TOPK), 5, device=dev, dtype=torch.int64); ids[:, 0] = gids[0]; ids[:, 1] = gids[1]
w = torch.zeros(T, TOPK, device=dev); w[:, 0] = 0.7; w[:, 1] = 0.4
ref = torch.zeros(T, H, device=dev)
for j, g in enumerate(gids):
    W1 = deq(c1[j], s1[j]).t(); W3 = deq(c3[j], s3[j]).t(); W2 = deq(c2[j], s2[j]).t()
    gate = (x.float() @ W1).clamp(max=10); up = (x.float() @ W3).clamp(-10, 10)
    ref += w[:, j:j+1] * ((F.silu(gate) * up) @ W2)
for low_first, gate_first in itertools.product([True, False], [True, False]):
    lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0,
                          dtype=torch.bfloat16, device=dev, decode_tokens=48, prefill_tokens=256)
    w13 = torch.empty(n_local, 2 * I, H // 2, dtype=torch.uint8, device=dev); w13s = torch.empty(n_local, 2 * I, H // 32, dtype=torch.uint8, device=dev)
    w2 = torch.empty(n_local, H, I // 2, dtype=torch.uint8, device=dev); w2s = torch.empty(n_local, H, I // 32, dtype=torch.uint8, device=dev)
    for j in range(n_local):
        g, u = pack(c1[j], low_first), pack(c3[j], low_first); gs, us = s1[j], s3[j]
        if not gate_first: g, u, gs, us = u, g, us, gs
        w13[j, :I], w13[j, I:], w13s[j, :I], w13s[j, I:] = g, u, gs, us
        w2[j], w2s[j] = pack(c2[j], low_first), s2[j]
    lay.build_keep(global_ids=gids, w13_fp4=w13, w13_scale=w13s, w2_fp4=w2, w2_scale=w2s)
    out = lay.forward(x, w, ids).float(); torch.cuda.synchronize()
    err = ((out - ref).norm() / ref.norm()).item(); cos = F.cosine_similarity(out.flatten(), ref.flatten(), 0).item()
    print(json.dumps(dict(low_nibble_first=low_first, gate_rows_first=gate_first, rel_err=round(err, 5), cos=round(cos, 6))), flush=True)
# scale-only probe: all codes = 1.0 (code 2), varying scales -> isolates scale handling
codes1 = torch.full((I, H), 2, device=dev, dtype=torch.uint8); codesd = torch.full((H, I), 2, device=dev, dtype=torch.uint8)
lay = Tr3LayerExperts(hidden_size=H, intermediate_size=I, route_num_experts=E, topk=TOPK, swiglu_limit=10.0, dtype=torch.bfloat16, device=dev, decode_tokens=48, prefill_tokens=256)
w13 = torch.cat([pack(codes1, True), pack(codes1, True)]).unsqueeze(0); w13s = torch.cat([s1[0], s3[0]]).unsqueeze(0); w2 = pack(codesd, True).unsqueeze(0); w2s = s2[0].unsqueeze(0)
lay.build_keep(global_ids=[gids[0]], w13_fp4=w13, w13_scale=w13s, w2_fp4=w2, w2_scale=w2s)
ids2 = torch.full((T, TOPK), 5, device=dev, dtype=torch.int64); ids2[:, 0] = gids[0]; w2w = torch.zeros(T, TOPK, device=dev); w2w[:, 0] = 1.0
out = lay.forward(x, w2w, ids2).float()
W1 = deq(codes1, s1[0]).t(); W3 = deq(codes1, s3[0]).t(); W2 = deq(codesd, s2[0]).t()
gate = (x.float() @ W1).clamp(max=10); up = (x.float() @ W3).clamp(-10, 10); ref2 = (F.silu(gate) * up) @ W2
print(json.dumps(dict(probe="scales_only_codes_1.0", rel_err=round(((out - ref2).norm() / ref2.norm()).item(), 5), cos=round(F.cosine_similarity(out.flatten(), ref2.flatten(), 0).item(), 6))), flush=True)
