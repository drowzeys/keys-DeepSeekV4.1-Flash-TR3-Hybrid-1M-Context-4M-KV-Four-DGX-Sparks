"""Runtime capture of post-wo_b last-token activations for refusal direction extraction."""
from __future__ import annotations
import os, re, time
from pathlib import Path
import torch

_LAYER_RE = re.compile(r"layers\.(\d+)")
_ENABLED = os.environ.get("ABLIT_CAPTURE", "0") == "1"
_DIR = Path(os.environ.get("ABLIT_CAPTURE_DIR", "/ablit_capture"))
_TAG = os.environ.get("ABLIT_CAPTURE_TAG", "run")
_TAG_FILE = Path(os.environ.get("ABLIT_CAPTURE_TAG_FILE", "/ablit_capture/CURRENT_TAG"))
_SEQ = 0
_BUF: dict[int, list] = {}
_LAST_WRITE = 0.0


def _current_tag() -> str:
    try:
        if _TAG_FILE.exists():
            s = _TAG_FILE.read_text().strip()
            if s:
                return s
    except Exception:
        pass
    return _TAG


def _rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank()) == 0
    except Exception:
        return True


def layer_id(layer_name: str) -> int | None:
    m = _LAYER_RE.search(layer_name or "")
    return int(m.group(1)) if m else None


def maybe_capture(out: torch.Tensor, layer_name: str) -> None:
    """out: [num_tokens, hidden] post-wo_b (after all-reduce)."""
    global _SEQ, _LAST_WRITE
    if not _ENABLED or out is None or out.numel() == 0:
        return
    if not _rank0():
        return
    lid = layer_id(layer_name)
    if lid is None:
        return
    vec = out[-1].detach().float().cpu()
    _BUF.setdefault(lid, []).append(vec)
    flush()
    _LAST_WRITE = time.time()


def flush() -> None:
    if not _BUF:
        return
    dest = _DIR / _current_tag()
    dest.mkdir(parents=True, exist_ok=True)
    for lid, vecs in list(_BUF.items()):
        if not vecs:
            continue
        path = dest / f"layer_{lid:02d}.pt"
        if path.exists():
            prev = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(prev, torch.Tensor):
                prev = [prev]
            elif not isinstance(prev, list):
                prev = list(prev)
        else:
            prev = []
        prev.extend(vecs)
        torch.save(prev, path)
        _BUF[lid] = []
    (dest / "FLUSHED").write_text(str(time.time()))
