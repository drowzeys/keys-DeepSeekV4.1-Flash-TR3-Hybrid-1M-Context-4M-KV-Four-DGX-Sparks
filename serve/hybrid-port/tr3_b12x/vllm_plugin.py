"""vLLM integration for TR3-Hybrid routed experts on B12X (see core.py).

Activated by sitecustomize (TR3_HYBRID=1). Patches:
  * DeepseekV4FP8Config.get_quant_method -> Tr3HybridMoEMethod for the 384-expert
    text layers (draft/DSpark experts (128) and everything else stay native).
  * DeepseekV4Model.load_weights (v4.1 nvidia) -> routes layers.L.ffn.experts.E.wX.{trellis,
    suh,svh,mcg} straight to the owning RoutedExperts; keep experts (w*.weight/.scale)
    flow through vLLM's standard expert mapping into our w13_/w2_ params.
No silent fallbacks: every local expert must receive every tensor or loading fails.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .core import K_WORDS, TILE, Tr3LayerExperts

logger = init_logger("vllm.tr3_b12x")   # must live under vllm.* to be emitted
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_TRELLIS_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mcg)$")
_KEEP_IDS: dict[int, list[int]] | None = None
_SHARED: dict[str, torch.Tensor] = {}   # shared B12X scratch across layers/tiers, per plan kind


def _heap_probe(n: int) -> None:
    """Is the anon growth live CPU tensors (retention) or unreturned freed heap (glibc)?"""
    import ctypes, gc
    try:
        before = _rss_anon()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        after = _rss_anon()
        ts = [o for o in gc.get_objects() if torch.is_tensor(o) and o.device.type == "cpu" and o.numel() > 65536]
        tot = sum(t.numel() * t.element_size() for t in ts) / 2**30
        ts.sort(key=lambda t: -t.numel() * t.element_size())
        who = []
        for t in ts[:3]:
            refs = [type(r).__name__ for r in gc.get_referrers(t) if not isinstance(r, list) or len(r) < 100000][:6]
            who.append(f"{tuple(t.shape)}:{t.dtype}<-{refs}")
        logger.info("TR3 heap probe @%d: RssAnon %.1fG -> %.1fG after malloc_trim; live CPU tensors>64K: %d (%.1f GiB); top: %s",
                    n, before, after, len(ts), tot, " | ".join(who))
    except Exception as exc:  # noqa: BLE001
        logger.info("TR3 heap probe failed: %r", exc)


def _rss_anon() -> float:
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("RssAnon"):
                return int(l.split()[1]) / 2**20
    return float("nan")


def _mem() -> str:
    parts = []
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
        parts.append(f"host_avail={mi['MemAvailable']/2**20:.1f}G")
        parts.append("meminfo[" + " ".join(f"{k}={mi.get(k,0)/2**20:.1f}G" for k in
                     ("MemFree", "Cached", "Mapped", "Shmem", "AnonPages", "Unevictable", "Mlocked", "KernelStack", "PageTables", "Slab")) + "]")
        with open("/proc/self/status") as f:
            st = {l.split(":")[0]: l.split()[1] for l in f if l.startswith(("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmPin", "VmLck"))}
        parts.append("self[" + " ".join(f"{k}={int(v)/2**20:.1f}G" for k, v in st.items()) + "]")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"meminfo_err={exc!r}")
    parts.append(f"gpu_alloc={torch.cuda.memory_allocated()/2**30:.1f}G" if torch.cuda.is_available() else "gpu=n/a")
    return " ".join(parts)


def keep_ids_by_layer(model_dir: str) -> dict[int, list[int]]:
    global _KEEP_IDS
    if _KEEP_IDS is None:
        idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        out: dict[int, set[int]] = {}
        for n in idx:
            m = re.match(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight$", n)
            if m:
                out.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        _KEEP_IDS = {k: sorted(v) for k, v in out.items()}
        logger.info("TR3: keep-expert ids read for %d layers from %s", len(_KEEP_IDS), model_dir)
    return _KEEP_IDS


class Tr3HybridMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, layer_idx: int, keep_ids: list[int], rank: int, world: int):
        super().__init__(moe)
        self.layer_idx, self.rank, self.world = layer_idx, rank, world
        E = moe.num_experts
        per = E // world
        assert per * world == E, (E, world)
        self.local = list(range(rank * per, (rank + 1) * per))
        keep = set(keep_ids)
        self.keep_local = [e for e in self.local if e in keep]
        self.tail_local = [e for e in self.local if e not in keep]
        self.tail_slot = {g: i for i, g in enumerate(self.tail_local)}
        self.keep_slot = {g: i for i, g in enumerate(self.keep_local)}

    # vLLM contract -------------------------------------------------------
    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra):
        H, I = int(hidden_size), int(self.moe.intermediate_size)   # FULL width, not the TP slice
        Et, Ek = len(self.tail_local), len(self.keep_local)
        attrs = {k: v for k, v in extra.items() if k != "weight_loader"}
        def reg(name, t):
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)
            set_weight_attrs(p, {"weight_loader": self.load_expert_tensor, "_tr3_name": name, "_tr3_layer": layer})
        # No explicit device: like every stock method, follow the caller's device
        # context (a meta-device construction pass must NOT allocate real memory).
        reg("w13_trellis", torch.empty(2, Et, H // TILE, I // TILE, K_WORDS, dtype=torch.int16))
        reg("w2_trellis", torch.empty(Et, I // TILE, H // TILE, K_WORDS, dtype=torch.int16))
        reg("gate_suh", torch.empty(Et, H, dtype=torch.float16))
        reg("up_suh", torch.empty(Et, H, dtype=torch.float16))
        reg("down_svh", torch.empty(Et, H, dtype=torch.float16))
        reg("inter_rot", torch.empty(Et, 3 * I, dtype=torch.float16))   # gate_svh|up_svh|down_suh
        reg("mcg", torch.zeros(Et, 3, dtype=torch.int32))
        reg("w13_weight", torch.empty(Ek, 2 * I, H // 2, dtype=torch.uint8))
        reg("w13_weight_scale", torch.empty(Ek, 2 * I, H // 32, dtype=torch.uint8))
        reg("w2_weight", torch.empty(Ek, H, I // 2, dtype=torch.uint8))
        reg("w2_weight_scale", torch.empty(Ek, H, I // 32, dtype=torch.uint8))
        layer._tr3_H, layer._tr3_I = H, I
        layer._tr3_loaded = {("tail", g): set() for g in self.tail_local} | {("keep", g): set() for g in self.keep_local}
        layer._tr3_experts = None
        logger.info("TR3 layer %d rank %d/%d: %d tail + %d keep experts (%d local of %d); params on %s; %s",
                    self.layer_idx, self.rank, self.world, Et, Ek, len(self.local), num_experts,
                    layer.w13_trellis.device, _mem())

    def load_expert_tensor(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        """Called with GLOBAL expert ids. Returns False (skip) for experts on other ranks."""
        global _LOAD_CALLS
        _LOAD_CALLS += 1
        if _LOAD_CALLS % 4000 == 0:
            logger.info("TR3 load progress: %d expert-tensor calls, last %s; %s", _LOAD_CALLS, weight_name, _mem())
        layer = param._tr3_layer
        g = int(expert_id)
        name = param._tr3_name
        kind = weight_name.rsplit(".", 1)[-1]
        t = loaded_weight
        if name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            kind = "weight_scale" if name.endswith("_scale") else "weight"
            if g not in self.keep_slot:
                return False if return_success else None
            s = self.keep_slot[g]
            if t.dtype != torch.uint8:
                t = t.view(torch.uint8)
            I = layer._tr3_I
            if name.startswith("w13"):
                rows = slice(0, I) if shard_id == "w1" else slice(I, 2 * I)
                assert shard_id in ("w1", "w3"), shard_id
                param.data[s, rows].copy_(t)
            else:
                assert shard_id == "w2", shard_id
                param.data[s].copy_(t)
            layer._tr3_loaded[("keep", g)].add(f"{shard_id}.{kind}")
            return True if return_success else None
        # trellis tier
        if g not in self.tail_slot:
            return False if return_success else None
        s = self.tail_slot[g]
        I = layer._tr3_I
        if kind == "trellis":
            if shard_id in ("w1", "w3"):
                layer.w13_trellis.data[0 if shard_id == "w1" else 1, s].copy_(t)
            else:
                layer.w2_trellis.data[s].copy_(t)
        elif kind == "suh":
            {"w1": layer.gate_suh, "w3": layer.up_suh}[shard_id].data[s].copy_(t) if shard_id != "w2" else \
                layer.inter_rot.data[s, 2 * I:].copy_(t)
        elif kind == "svh":
            if shard_id == "w1":
                layer.inter_rot.data[s, :I].copy_(t)
            elif shard_id == "w3":
                layer.inter_rot.data[s, I:2 * I].copy_(t)
            else:
                layer.down_svh.data[s].copy_(t)
        elif kind == "mcg":
            layer.mcg.data[s, {"w1": 0, "w3": 1, "w2": 2}[shard_id]] = int(t.reshape(-1)[0])
        else:
            raise ValueError(f"unknown TR3 tensor kind {kind} in {weight_name}")
        layer._tr3_loaded[("tail", g)].add(f"{shard_id}.{kind}")
        return True if return_success else None

    def process_weights_after_loading(self, layer):
        need_tail = {f"{w}.{k}" for w in ("w1", "w3", "w2") for k in ("trellis", "suh", "svh", "mcg")}
        need_keep = {f"{w}.{k}" for w in ("w1", "w3", "w2") for k in ("weight", "weight_scale")}
        missing = [(t, g, sorted((need_tail if t == "tail" else need_keep) - got))
                   for (t, g), got in layer._tr3_loaded.items() if (need_tail if t == "tail" else need_keep) - got]
        if missing:
            raise RuntimeError(f"TR3 layer {self.layer_idx}: {len(missing)} experts incomplete, e.g. {missing[:3]}")
        dev = layer.w13_trellis.device
        if dev.type != "cuda":
            raise RuntimeError(f"TR3 layer {self.layer_idx}: params on {dev}, expected cuda after load")
        exp = Tr3LayerExperts(hidden_size=layer._tr3_H, intermediate_size=layer._tr3_I,
                              route_num_experts=self.moe.num_experts, topk=self.moe.experts_per_token,
                              swiglu_limit=float(self.moe.swiglu_limit or 10.0), dtype=self.moe.in_dtype,
                              device=dev, decode_tokens=max(int(self.moe.max_capture_size), 8),
                              prefill_tokens=int(self.moe.max_num_tokens), shared_scratch=_SHARED)
        exp.set_local_ids(self.local)
        exp.layer_idx = self.layer_idx
        exp.build_tail(global_ids=self.tail_local, w13_trellis=layer.w13_trellis.data, w2_trellis=layer.w2_trellis.data,
                       gate_suh=layer.gate_suh.data, up_suh=layer.up_suh.data, intermediate_rotations=layer.inter_rot.data,
                       down_svh=layer.down_svh.data, mcg=layer.mcg.data)
        exp.build_keep(global_ids=self.keep_local, w13_fp4=layer.w13_weight.data, w13_scale=layer.w13_weight_scale.data,
                       w2_fp4=layer.w2_weight.data, w2_scale=layer.w2_weight_scale.data)
        # keep-tier bytes were repacked into B12X's own buffers: free the checkpoint-layout copies
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            getattr(layer, n).data = torch.empty(0, dtype=torch.uint8, device=dev)
        if os.environ.get("TR3_PREWARM", "1") == "1":
            exp.prewarm()
        layer._tr3_experts = exp
        logger.info("TR3 layer %d ready: %d tail + %d keep experts, decode<=%d prefill<=%d tokens; %s",
                    self.layer_idx, len(self.tail_local), len(self.keep_local),
                    exp.sizes["decode"][0], exp.sizes["prefill"][0], _mem())

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        return layer._tr3_experts.forward(x, topk_weights, topk_ids)

    def apply_monolithic(self, *a, **k):
        raise NotImplementedError("TR3 method is modular")


_LAYERS: dict[int, torch.nn.Module] = {}   # layer_idx -> RoutedExperts (text layers)
_LOAD_CALLS = 0
_PATCHED = False


def register() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
    from vllm.models.deepseek_v4_1.nvidia import model as v41

    model_dir = os.environ.get("TR3_MODEL_DIR", "")
    orig_gqm = DeepseekV4FP8Config.get_quant_method

    def get_quant_method(self, layer, prefix=""):
        if isinstance(layer, RoutedExperts) and layer.moe_config.num_experts == 384:
            m = _LAYER_RE.search(prefix or layer.layer_name or "")
            if m is not None and int(m.group(1)) < 40:
                li = int(m.group(1))
                keep = keep_ids_by_layer(model_dir).get(li, [])
                method = Tr3HybridMoEMethod(layer.moe_config, li, keep,
                                            get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
                _LAYERS[li] = layer
                return method
        return orig_gqm(self, layer, prefix)
    get_quant_method._tr3 = True
    DeepseekV4FP8Config.get_quant_method = get_quant_method

    orig_load = v41.DeepseekV4Model.load_weights

    def load_weights(self, weights):
        extra_loaded: set[str] = set()
        by_layer = {}
        for n, mod in self.named_modules():
            if isinstance(mod, RoutedExperts) and isinstance(mod.quant_method, Tr3HybridMoEMethod):
                by_layer[mod.quant_method.layer_idx] = (n, mod)

        def gen():
            n = 0
            for name, tensor in weights:
                n += 1
                if n % 10000 == 0:
                    logger.info("TR3 load stream: %d tensors seen, at %s; %s", n, name, _mem())
                    _heap_probe(n)
                m = _TRELLIS_RE.match(name)
                if m is None:
                    yield name, tensor
                    continue
                li, e, shard, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                if li not in by_layer:
                    raise RuntimeError(f"TR3 tensor {name} for a layer without the TR3 method")
                modname, mod = by_layer[li]
                pname = {"trellis": "w13_trellis" if shard != "w2" else "w2_trellis",
                         "suh": {"w1": "gate_suh", "w3": "up_suh", "w2": "inter_rot"}[shard],
                         "svh": {"w1": "inter_rot", "w3": "inter_rot", "w2": "down_svh"}[shard],
                         "mcg": "mcg"}[kind]
                param = getattr(mod, pname)
                ok = param.weight_loader(param, tensor, name, shard_id=shard, expert_id=e, return_success=True)
                if ok:
                    extra_loaded.add(f"{modname}.{pname}")
        loaded = set(orig_load(self, gen()))
        # every TR3 param name counts as loaded once any expert landed in it (+ empty tiers)
        for li, (modname, mod) in by_layer.items():
            for pname in ("w13_trellis", "w2_trellis", "gate_suh", "up_suh", "down_svh", "inter_rot", "mcg",
                          "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
                loaded.add(f"{modname}.{pname}")
        return loaded | extra_loaded
    load_weights._tr3 = True
    v41.DeepseekV4Model.load_weights = load_weights

    # DEBUG: fingerprint a few non-expert weights after load; core.forward calls
    # _check_fingerprints() when it sees the first non-finite output.
    if os.environ.get("TR3_DEBUG") == "1":
        from . import core as _core
        orig_pw = v41.DeepseekV41LLMForCausalLM.process_weights_after_loading

        def pw(self):
            orig_pw(self)
            fp = {}
            import itertools
            for name, p in itertools.chain(self.named_parameters(), (("BUF:" + n, b) for n, b in self.named_buffers())):
                d = p.data if hasattr(p, "data") else p
                if d.numel() == 0 or d.device.type != "cuda":
                    continue
                s = float(d.float().sum()) if d.dtype != torch.float8_e8m0fnu else float(d.view(torch.uint8).float().sum())
                fp[name] = (s, int(d.numel()), int(d.data_ptr()), int(d.numel() * d.element_size()))
            _core.FINGERPRINTS = fp
            _core.FINGERPRINT_MODEL = self
            bufs = []
            for k, b in _SHARED.items():
                bufs.append((f"shared_scratch[{k}]", int(b.data_ptr()), int(b.numel() * b.element_size())))
            for li, lay in _LAYERS.items():
                exp = getattr(lay, "_tr3_experts", None)
                if exp is None:
                    continue
                for tn, tier in (("tail", exp.tail), ("keep", exp.keep)):
                    if tier is None:
                        continue
                    for k, ob in tier.outbuf.items():
                        bufs.append((f"L{li}.{tn}.outbuf[{k}]", int(ob.data_ptr()), int(ob.numel() * ob.element_size())))
            _core.PLUGIN_BUFS = bufs
            logger.info("TR3 DEBUG: fingerprinted %d parameter tensors; %d plugin buffers mapped", len(fp), len(bufs))
            # Per-rank, pre-all-reduce probes on layer 0 (and the embedding): log non-finite
            # counts of inputs/outputs for eager forwards with 2 < T <= 64 (real short prefills).
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            state = {"n": 0}

            def make_hook(name):
                def hook(mod, inputs, output):
                    if torch.cuda.is_current_stream_capturing():
                        return
                    def stats(t):
                        if not torch.is_tensor(t) or t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                            return "-"
                        f = t.detach().float()
                        return f"nonfinite={int((~torch.isfinite(f)).sum())}/{f.numel()} absmax={f.abs().max().item() if torch.isfinite(f).all() else float('nan'):.3g} shape={tuple(t.shape)}"
                    ins = [i for i in inputs if torch.is_tensor(i)]
                    T = ins[0].shape[0] if ins and ins[0].dim() >= 1 else -1
                    if not (2 < T <= 64):
                        return
                    # skip vLLM's dummy/warmup forwards (all-zero float inputs or zero ids)
                    def _allzero(t):
                        if not torch.is_tensor(t) or t.numel() == 0:
                            return True
                        return bool((t == 0).all())
                    if all(_allzero(i) for i in ins[:2]):
                        return
                    state["n"] += 1
                    if state["n"] > 400:
                        return
                    outs = output if isinstance(output, (tuple, list)) else (output,)
                    extra = ""
                    kv = getattr(mod, "kv_cache", None)
                    if torch.is_tensor(kv) and kv.numel() > 0 and kv.dtype in (torch.bfloat16, torch.float16, torch.float32):
                        nf = int((~torch.isfinite(kv.float())).sum()) if kv.numel() < (1 << 31) else -1
                        extra = f" kv_cache[nonfinite={nf}/{kv.numel()} shape={tuple(kv.shape)} dtype={kv.dtype}]"
                    logger.error("TR3 PROBE rank%d %s: in[%s] out[%s]%s", rank, name,
                                 " | ".join(stats(i) for i in ins[:2]), " | ".join(stats(o) for o in outs[:2]), extra)
                return hook
            hooked = 0
            for name, mod in self.named_modules():
                if any(name.endswith(s) for s in ("embed_tokens", "layers.0.attn", "layers.0.attn.fused_wqa_wkv", "layers.0.attn.wq_b",
                                                   "layers.0.attn.wo_a", "layers.0.attn.wo_b", "layers.0.attn.q_norm", "layers.0.attn.kv_norm",
                                                   "layers.0.attn.indexer", "layers.0.attn.compressor", "layers.0.attn.rotary_emb",
                                                   "layers.0.ffn.gate", "layers.0.ffn.shared_experts", "layers.0.ffn", "layers.0", "layers.1")):
                    mod.register_forward_hook(make_hook(name)); hooked += 1
            logger.info("TR3 DEBUG: %d forward probes registered on rank %d", hooked, rank)
        v41.DeepseekV41LLMForCausalLM.process_weights_after_loading = pw

    # The VL wrapper does `sorted(mapper.apply(weights))`, materializing EVERY
    # checkpoint tensor as a live mmap view before loading (108k tensors / 426 GiB
    # of mappings here). With all sources pinned alive, host memory grew ~1 GB per
    # layer during loading and took the whole cluster down. Stream instead: all
    # language_model.* weights first (one contiguous group for AutoWeightsLoader,
    # so the child's finalization still runs on a complete model), then the few
    # non-language (vision/aligner) tensors that were deferred.
    from vllm.models.deepseek_v4_1.nvidia import vl_model as v41vl
    from vllm.model_executor.models.utils import AutoWeightsLoader

    def vl_load_weights(self, weights):
        mapped = self.hf_to_vllm_mapper.apply(weights)
        deferred: list = []

        def stream():
            for name, t in mapped:
                if name.startswith("language_model."):
                    yield name, t
                else:
                    deferred.append((name, t))
            logger.info("TR3: language_model weights streamed; %d deferred non-language tensors", len(deferred))
            yield from deferred
        loaded = AutoWeightsLoader(self).load_weights(stream())
        self._weights_finalized = True
        return loaded
    vl_load_weights._tr3 = True
    v41vl.DeepseekV41ForCausalLM.load_weights = vl_load_weights
    logger.info("TR3 hybrid B12X plugin registered (model_dir=%s); VL load_weights streams (no sorted() materialization)", model_dir)

    if os.environ.get("ABLIT_CAPTURE") == "1":
        from vllm.models.deepseek_v4_1.nvidia import flashinfer_sparse as _fi
        import ablit_capture as _ac

        def _wrap_o_proj(cls):
            orig = cls._o_proj
            if getattr(orig, "_ablit_wrapped", False):
                return

            def _o_proj(self, o, positions, _orig=orig):
                out = _orig(self, o, positions)
                try:
                    name = getattr(self, "prefix", "") or getattr(self, "layer_name", "") or ""
                    _ac.maybe_capture(out, name)
                except Exception:
                    pass
                return out

            _o_proj._ablit_wrapped = True
            cls._o_proj = _o_proj

        wrapped = []
        for _cls_name in ("DeepseekV4FlashInferMLAAttention", "DeepseekV4FlashInferSM120Attention"):
            _cls = getattr(_fi, _cls_name, None)
            if _cls is not None:
                _wrap_o_proj(_cls)
                wrapped.append(_cls_name)
        logger.info("TR3 ABLIT_CAPTURE wrapped _o_proj on %s", wrapped or "NONE")
