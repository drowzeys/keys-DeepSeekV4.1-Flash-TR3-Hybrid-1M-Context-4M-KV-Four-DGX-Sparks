"""TR3-Hybrid routed experts on Brandon B12X W4A16 kernels (GB10 / SM121).

One rank owns a contiguous slice of the 384 experts at FULL width (expert
parallel inside TP: every rank routes all tokens, computes only its own
experts, and vLLM's TP all-reduce sums the partial routed outputs).
Two tiers per layer, both through B12X:
  * tail : EXL3 K3 trellis experts (btx source, full-rotation fused kernel,
           swiglu clamp applied inside the rotation epilogue -> patched kernel)
  * keep : native DeepSeek MXFP4 experts (fp4_e8m0_k32 source, packed layout)
Experts outside the rank map to -1 and contribute nothing.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Sequence

import torch

log = logging.getLogger(__name__)
MCG_MARKER = 0xCBAC1FED
K_BITS = 3
TILE = 16
K_WORDS = K_BITS * TILE  # 48 int16 per 16x16 tile at 3 bpw
import os
KEEP_W13_LAYOUT = os.environ.get("TR3_KEEP_W13_LAYOUT", "w31")
DEBUG = os.environ.get("TR3_DEBUG", "0") == "1"
SKIP_IN_CAPTURE = os.environ.get("TR3_SKIP_IN_CAPTURE", "0") == "1"   # bisect: return zeros while capturing
NO_DECODE_PLAN = os.environ.get("TR3_NO_DECODE_PLAN", "0") == "1"   # bisect: prefill plan (block_m 32) for everything
FAST_MATH = os.environ.get("TR3_FAST_MATH", "0") == "1"        # speed lever: B12X fast_math path in bind
_ENV_DEC_BM = os.environ.get("TR3_DECODE_BLOCK_M")             # override decode w4a16 block_m (default 8)
_ENV_PRE_BM = os.environ.get("TR3_PREFILL_BLOCK_M")            # override prefill w4a16 block_m (default 32)
FINGERPRINTS: dict = {}
FINGERPRINT_MODEL = None
_FP_CHECKED = False
PLUGIN_BUFS: list = []


_FP_CALLS = 0


def _check_fingerprints(tag: str = "") -> None:
    """Re-check all parameter fingerprints; log changes and re-baseline."""
    global _FP_CALLS
    if FINGERPRINT_MODEL is None:
        return
    _FP_CALLS += 1
    changed = []
    import itertools
    for name, p in itertools.chain(FINGERPRINT_MODEL.named_parameters(), (("BUF:" + n, b) for n, b in FINGERPRINT_MODEL.named_buffers())):
        if name in FINGERPRINTS and p.numel():
            d = p.data if hasattr(p, "data") else p
            s = float(d.float().sum()) if d.dtype != torch.float8_e8m0fnu else float(d.view(torch.uint8).float().sum())
            s0, n, ptr, nbytes = FINGERPRINTS[name]
            if s != s0 or s != s:
                changed.append(f"{name} {s0:.6g}->{s:.6g}")
                FINGERPRINTS[name] = (s, n, ptr, nbytes)
    if changed:
        log.error("TR3 DEBUG fingerprints CHANGED at check #%d (%s): %d tensors: %s", _FP_CALLS, tag, len(changed), changed[:6])
    elif _FP_CALLS <= 12 or _FP_CALLS % 50 == 0:
        log.error("TR3 DEBUG fingerprints unchanged at check #%d (%s)", _FP_CALLS, tag)


def _api():
    from b12x.moe import fused_moe as api
    return api


@dataclasses.dataclass
class TierPlans:
    experts: object                     # B12XFP4ExpertWeights
    weight_plan: object
    plans: dict[str, object]            # name -> TPMoEScratchPlan
    scratch: dict[str, object]
    expert_map: torch.Tensor            # int32 [route_num_experts] -> local slot or -1
    max_tokens: dict[str, int]
    full_rotation: bool = True          # trellis plans accept output_expert_map; packed plans do not
    outbuf: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)  # caller-owned outputs (graph capture)

    def run(self, x, topk_weights, topk_ids, kind):
        api = _api()
        kw = dict(route_expert_map=self.expert_map)
        if self.full_rotation:
            kw["output_expert_map"] = self.expert_map
        m = int(x.shape[0])
        scratch = self.scratch[kind]
        if isinstance(scratch, _SharedRef):
            scratch = scratch.get()
        out = self.outbuf[kind][:m]
        # Rows with no routed LOCAL expert are not written by the kernel; the
        # caller-owned buffer would otherwise hand back stale rows. Zero first.
        out.zero_()
        if FAST_MATH:
            kw["fast_math"] = True
        binding = api.bind(
            self.plans[kind], scratch=scratch, a=x, experts=self.experts,
            topk_weights=topk_weights, topk_ids=topk_ids, output=out, **kw)
        return api.run(binding=binding)


class _SharedRef:
    """Resolves the (possibly re-grown) shared scratch buffer at bind time."""
    def __init__(self, pool, kind):
        self.pool, self.kind = pool, kind
    def get(self):
        return [self.pool[self.kind]]


class Tr3LayerExperts:
    """GPU state for one layer's local experts (tail + keep) on one rank."""

    def __init__(self, *, hidden_size: int, intermediate_size: int, route_num_experts: int,
                 topk: int, swiglu_limit: float, dtype: torch.dtype, device: torch.device,
                 decode_tokens: int, prefill_tokens: int, decode_block_m: int = 8,
                 prefill_block_m: int = 32, shared_scratch: dict | None = None):
        self.shared_scratch = shared_scratch   # kind -> one uint8 buffer big enough for every layer/tier
        self.H, self.I, self.E = hidden_size, intermediate_size, route_num_experts
        # B12X native BTX (trellis) weights require fp16 GEMM operands; run both
        # tiers in fp16 and hand back `dtype` (the model's bf16) to the caller.
        self.topk, self.limit, self.out_dtype, self.device = topk, float(swiglu_limit), dtype, device
        self.dtype = torch.float16
        if _ENV_DEC_BM: decode_block_m = int(_ENV_DEC_BM)
        if _ENV_PRE_BM: prefill_block_m = int(_ENV_PRE_BM)
        self.sizes = {"decode": (int(decode_tokens), decode_block_m),
                      "prefill": (int(prefill_tokens), prefill_block_m)}
        self.tail: TierPlans | None = None
        self.keep: TierPlans | None = None
        self.nonlocal_id = 0   # set by set_local_ids(): a global id this rank does not own

    # ---- planning helpers -------------------------------------------------
    def _plans_for(self, weight_plan, experts, expert_map, full_rotation=True) -> TierPlans:
        api = _api()
        plans, scratch, maxt, outbuf = {}, {}, {}, {}
        for kind, (m, bm) in self.sizes.items():
            caps = api.Caps(max_tokens=m, num_topk=self.topk, device=self.device,
                            weight_plan=weight_plan, quant_mode="w4a16",
                            route_num_experts=self.E, w4a16_block_size_m=bm,
                            swiglu_limit=self.limit)
            plan = api.plan(caps)
            specs = plan.scratch_specs()
            assert len(specs) == 1, "B12X TP MoE plans expose one scratch buffer"
            spec = specs[0]
            if self.shared_scratch is not None:
                cur = self.shared_scratch.get(kind)
                if cur is None or cur.numel() < spec.shape[0] or cur.dtype != spec.dtype:
                    self.shared_scratch[kind] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                bufs = _SharedRef(self.shared_scratch, kind)
            else:
                bufs = [torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)]
            plans[kind], scratch[kind], maxt[kind] = plan, bufs, m
            # full-rotation trellis plans write fp32 output; packed plans write params dtype
            outbuf[kind] = torch.empty(m, self.H, dtype=torch.float32 if full_rotation else self.dtype,
                                       device=self.device)
        return TierPlans(experts=experts, weight_plan=weight_plan, plans=plans, scratch=scratch,
                         expert_map=expert_map, max_tokens=maxt, full_rotation=full_rotation, outbuf=outbuf)

    def set_local_ids(self, local_ids) -> None:
        owned = set(int(g) for g in local_ids)
        self.nonlocal_id = next(g for g in range(self.E) if g not in owned)

    @staticmethod
    def make_expert_map(route_num_experts: int, global_ids: Sequence[int], device) -> torch.Tensor:
        m = torch.full((route_num_experts,), -1, dtype=torch.int32)
        for slot, g in enumerate(global_ids):
            m[int(g)] = slot
        return m.to(device)

    def build_tail(self, *, global_ids: Sequence[int], w13_trellis: torch.Tensor, w2_trellis: torch.Tensor,
                   gate_suh: torch.Tensor, up_suh: torch.Tensor, intermediate_rotations: torch.Tensor,
                   down_svh: torch.Tensor, mcg: torch.Tensor) -> None:
        """w13_trellis int16 [2,E,H/16,I/16,48]; w2_trellis [E,I/16,H/16,48]; suh/svh fp16 [E,H];
        intermediate_rotations fp16 [E,3I] = [gate_svh|up_svh|down_suh]; mcg int32 [E,3]."""
        api = _api()
        from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
        E = len(global_ids)
        if E == 0:
            self.tail = None
            return
        assert w13_trellis.shape == (2, E, self.H // TILE, self.I // TILE, K_WORDS), w13_trellis.shape
        assert w2_trellis.shape == (E, self.I // TILE, self.H // TILE, K_WORDS), w2_trellis.shape
        bad = (mcg.to(torch.int64) & 0xFFFFFFFF) != MCG_MARKER
        if bool(bad.any()):
            raise RuntimeError(f"{int(bad.sum())} trellis tensors lack the MCG marker (unloaded expert?)")
        tile = api.select_w4a16_full_rotation_tile_config(
            hidden_size=self.H, intermediate_size=self.I, num_experts=E, topk=self.topk,
            route_num_experts=self.E, trellis_bits=K_BITS, use_expert_map=True, device=self.device)
        prepared = prepare_trellis256_moe_weights(
            w13=w13_trellis, w2=w2_trellis, hidden_size=self.H, intermediate_size=self.I,
            num_experts=E, activation="silu", fc1_tile_n=tile[1], fc2_tile_n=tile[3],
            params_dtype=self.dtype, w13_layout="trellis_t256_proj", trellis_bits=K_BITS,
            codebook="mcg", gate_suh=gate_suh, up_suh=up_suh,
            intermediate_rotations=intermediate_rotations, down_svh=down_svh, tile_config=tile)
        wp = api.plan_weights(quant_modes="w4a16", source_format="btx", activation="silu",
                              params_dtype=self.dtype, num_experts=E, hidden_size=self.H,
                              intermediate_size=self.I, w13_layout="w13", trellis_bits=K_BITS,
                              trellis_tile_config=tile, trellis_codebook="mcg",
                              trellis_rate_structure="uniform")
        experts = api.adopt_btx_weights(plan=wp, prepared=prepared)
        self.tail = self._plans_for(wp, experts, self.make_expert_map(self.E, global_ids, self.device))

    def build_keep(self, *, global_ids: Sequence[int], w13_fp4: torch.Tensor, w13_scale: torch.Tensor,
                   w2_fp4: torch.Tensor, w2_scale: torch.Tensor) -> None:
        """Native MXFP4 (e2m1 x2 per byte, e8m0 K/32 scales). w13_fp4 uint8 [E,2I,H/2],
        w13_scale uint8 [E,2I,H/32], w2_fp4 [E,H,I/2], w2_scale [E,H,I/32]. Rows 0:I gate, I:2I up."""
        api = _api()
        E = len(global_ids)
        if E == 0:
            self.keep = None
            return
        assert w13_fp4.shape == (E, 2 * self.I, self.H // 2), w13_fp4.shape
        assert w2_fp4.shape == (E, self.H, self.I // 2), w2_fp4.shape
        # B12X naming: "w13" = rows [up; gate] (needs a swap), "w31" = kernel-native [gate; up].
        # Our stacked rows are [gate(w1); up(w3)] -> "w31".
        layout = KEEP_W13_LAYOUT
        kp = api.plan_weights(quant_modes="w4a16", source_format="fp4_e8m0_k32", activation="silu",
                              params_dtype=self.dtype, num_experts=E, hidden_size=self.H,
                              intermediate_size=self.I, w13_layout=layout)
        ones = torch.ones(E, dtype=torch.float32, device=self.device)
        experts = api.prepare_weights(plan=kp, params_dtype=self.dtype,
                                      w1_fp4=w13_fp4, w1_blockscale=w13_scale, w1_global_scale=ones,
                                      w2_fp4=w2_fp4, w2_blockscale=w2_scale, w2_global_scale=ones,
                                      a1_gscale=ones, a2_gscale=ones)
        self.keep = self._plans_for(kp, experts, self.make_expert_map(self.E, global_ids, self.device),
                                    full_rotation=False)

    def prewarm(self) -> None:
        """Eagerly launch every plan of every tier once (all token counts that the
        decode plan can see, plus one prefill size) so kernels are compiled and
        first-launched OUTSIDE any CUDA graph capture."""
        torch.manual_seed(0)
        sizes = list(range(1, self.sizes["decode"][0] + 1)) + [min(64, self.sizes["prefill"][0])]
        for T in sizes:
            x = torch.randn(T, self.H, device=self.device).to(self.out_dtype)
            ids = torch.randint(0, self.E, (T, self.topk), device=self.device, dtype=torch.int32)
            w = torch.softmax(torch.randn(T, self.topk, device=self.device), -1)
            self.forward(x, w, ids)
        torch.cuda.synchronize()

    # ---- forward ------------------------------------------------------------
    def kind_for(self, num_tokens: int) -> str:
        if num_tokens <= self.sizes["decode"][0] and not NO_DECODE_PLAN:
            return "decode"
        if num_tokens <= self.sizes["prefill"][0]:
            return "prefill"
        raise ValueError(f"{num_tokens} tokens exceed planned prefill capacity {self.sizes['prefill'][0]}")

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """x [T,H] (self.dtype); topk_weights fp32 [T,k]; topk_ids [T,k] GLOBAL ids. Returns [T,H]
        partial output for this rank's experts (others contribute 0)."""
        kind = self.kind_for(x.shape[0])
        if SKIP_IN_CAPTURE and torch.cuda.is_current_stream_capturing():
            # bisect: keep B12X kernels OUT of captured graphs entirely
            return torch.zeros(x.shape, dtype=self.out_dtype, device=x.device)
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if topk_ids.dtype != torch.int64:
            topk_ids = topk_ids.to(torch.int64)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.float()
        # Padded / masked slots (id < 0 or >= E) -> a non-local expert id with weight 0,
        # so the expert maps resolve to -1 and nothing is indexed out of range.
        invalid = (topk_ids < 0) | (topk_ids >= self.E)
        topk_ids = torch.where(invalid, torch.full_like(topk_ids, self.nonlocal_id), topk_ids)
        topk_weights = torch.where(invalid, torch.zeros_like(topk_weights), topk_weights)
        out = None
        if DEBUG and getattr(self, "layer_idx", -1) == 0:
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                _check_fingerprints(f"layer0 pre-MoE T={x.shape[0]} kind={kind}")
            else:
                log.error("TR3 DEBUG layer0 forward under CUDA graph capture T=%d kind=%s", x.shape[0], kind)
        for tier in (self.tail, self.keep):
            if tier is None:
                continue
            y = tier.run(x, topk_weights, topk_ids, kind)   # view into the tier's own output buffer
            if DEBUG and not torch.cuda.is_current_stream_capturing():
                bad = int((~torch.isfinite(y)).sum())
                if bad:
                    _check_fingerprints(f"first NaN layer={getattr(self, 'layer_idx', '?')}")
                    log.error("TR3 DEBUG layer=%s tier=%s kind=%s T=%d nonfinite=%d x[absmax=%.1f mean=%.3f] w[max=%.3f] ids[min=%d max=%d]",
                              getattr(self, "layer_idx", "?"), "tail" if tier is self.tail else "keep", kind, x.shape[0], bad,
                              x.float().abs().max().item(), x.float().mean().item(), topk_weights.max().item(),
                              int(topk_ids.min()), int(topk_ids.max()))
            out = y if out is None else out + y
        if out is None:
            return torch.zeros(x.shape, dtype=self.out_dtype, device=x.device)
        return out.to(self.out_dtype)
