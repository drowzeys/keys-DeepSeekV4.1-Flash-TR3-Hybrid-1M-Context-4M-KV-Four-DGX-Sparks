import os, sys
if os.environ.get("TR3_HYBRID") == "1":
    try:
        import tr3_b12x.vllm_plugin as _p
        _p.register()
    except Exception as exc:  # loud, not silent
        print(f"[tr3_b12x] registration FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
