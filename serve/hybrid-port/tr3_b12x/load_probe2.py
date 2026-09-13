"""Bisect the retention: MODE=vllm uses vLLM's safetensors_weights_iterator + WeightsMapper; MODE=plain uses safe_open directly.
Copies local expert tensors into preallocated CUDA buffers exactly like the plugin; prints RssAnon every 10k tensors."""
import glob, json, os, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import TILE, K_WORDS
MODE = os.environ.get("MODE", "vllm"); R = 2; model = "/model"; H, I, E = 5120, 2304, 384; dev = torch.device("cuda:0")
def rss():
    with open("/proc/self/status") as f: st = {l.split(":")[0]: int(l.split()[1]) for l in f if l.startswith(("VmRSS", "RssAnon", "RssFile"))}
    with open("/proc/meminfo") as f: mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return f"RssAnon={st['RssAnon']/2**20:.1f}G RssFile={st['RssFile']/2**20:.1f}G avail={mi['MemAvailable']/2**20:.1f}G"
index = json.loads(open(f"{model}/model.safetensors.index.json").read())["weight_map"]
local = list(range(R * 96, (R + 1) * 96)); P = {}
for L in range(40):
    keep = [e for e in local if f"layers.{L}.ffn.experts.{e}.w1.weight" in index]; tail = [e for e in local if e not in keep]; Et, Ek = len(tail), len(keep)
    P[L] = dict(tail={g: i for i, g in enumerate(tail)}, keep={g: i for i, g in enumerate(keep)},
        w13t=torch.empty(2, Et, H // TILE, I // TILE, K_WORDS, dtype=torch.int16, device=dev), w2t=torch.empty(Et, I // TILE, H // TILE, K_WORDS, dtype=torch.int16, device=dev),
        gsuh=torch.empty(Et, H, dtype=torch.float16, device=dev), usuh=torch.empty(Et, H, dtype=torch.float16, device=dev), dsvh=torch.empty(Et, H, dtype=torch.float16, device=dev),
        ir=torch.empty(Et, 3 * I, dtype=torch.float16, device=dev), mcg=torch.zeros(Et, 3, dtype=torch.int32, device=dev),
        w13k=torch.empty(Ek, 2 * I, H // 2, dtype=torch.uint8, device=dev), w13s=torch.empty(Ek, 2 * I, H // 32, dtype=torch.uint8, device=dev),
        w2k=torch.empty(Ek, H, I // 2, dtype=torch.uint8, device=dev), w2s=torch.empty(Ek, H, I // 32, dtype=torch.uint8, device=dev))
print(f"[{MODE}] after create: {rss()}", flush=True)
files = sorted(glob.glob(f"{model}/model-*.safetensors"))
if MODE == "vllm":
    from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
    from vllm.models.deepseek_v4_1.nvidia.model import _make_deepseek_v4_weights_mapper
    it = _make_deepseek_v4_weights_mapper("fp4", "weight_scale").apply(safetensors_weights_iterator(files, use_tqdm_on_load=False))
    rx = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mcg|weight|weight_scale)$")
else:
    from safetensors import safe_open
    def plain():
        for sh in files:
            with safe_open(sh, framework="pt") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
    it = plain(); rx = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(trellis|suh|svh|mcg|weight|scale)$")
n = copied = 0; t0 = time.time()
for name, t in it:
    n += 1; m = rx.match(name)
    if m:
        L, g, w, k = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4); p = P[L]
        if g in p["tail"] and k in ("trellis", "suh", "svh", "mcg"):
            s = p["tail"][g]; copied += 1
            if k == "trellis": (p["w13t"][0 if w == "w1" else 1, s] if w != "w2" else p["w2t"][s]).copy_(t)
            elif k == "suh": ({"w1": p["gsuh"], "w3": p["usuh"]}[w][s] if w != "w2" else p["ir"][s, 2 * I:]).copy_(t)
            elif k == "svh": (p["ir"][s, :I] if w == "w1" else p["ir"][s, I:2 * I] if w == "w3" else p["dsvh"][s]).copy_(t)
            else: p["mcg"][s, {"w1": 0, "w3": 1, "w2": 2}[w]] = int(t.reshape(-1)[0])
        elif g in p["keep"] and k in ("weight", "scale", "weight_scale"):
            s = p["keep"][g]; copied += 1; u = t.view(torch.uint8)
            if k == "weight": (p["w13k"][s, :I] if w == "w1" else p["w13k"][s, I:] if w == "w3" else p["w2k"][s]).copy_(u)
            else: (p["w13s"][s, :I] if w == "w1" else p["w13s"][s, I:] if w == "w3" else p["w2s"][s]).copy_(u)
    if n % 10000 == 0:
        print(f"[{MODE}] {n} tensors, {copied} copied, {time.time()-t0:.0f}s: {rss()}", flush=True)
        if n >= 60000: break
print(f"[{MODE}] DONE {n} tensors {copied} copied: {rss()}", flush=True)
