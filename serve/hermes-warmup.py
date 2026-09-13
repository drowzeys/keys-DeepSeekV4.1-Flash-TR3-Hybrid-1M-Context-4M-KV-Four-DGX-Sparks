#!/usr/bin/env python3
"""Warm the 1M TR3 serve with a Hermes-shaped first request.

First real Hermes turn is ~25k-char system + ~23 tool schemas (~40k chars).
Without this, CUDA-graph / DSpark capture on that shape makes Telegram
look dead (20–200s TTFT). Run after /v1/models is up, before pointing Hermes.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def _tools(n: int = 20) -> list[dict]:
    tools = []
    for i in range(n):
        tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i:02d}",
                "description": ("A Hermes-visible tool schema used only to match first-turn "
                                "prefill size. " * 8),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string", "description": "x" * 120},
                        "path": {"type": "string", "description": "y" * 120},
                    },
                    "required": ["arg"],
                },
            },
        })
    return tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://10.100.10.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    args = ap.parse_args()
    system = (
        "You are Hermes Agent. Never list skills catalogs. "
        "Greetings: one short sentence, no tools. " * 80
    )
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "ping"},
        ],
        "tools": _tools(),
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    data = json.dumps(body).encode()
    print(f"warmup prompt_chars={len(system)} tools={len(body['tools'])} body={len(data)}", flush=True)
    t0 = time.time()
    req = urllib.request.Request(
        args.api.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read())
    dt = time.time() - t0
    text = (j["choices"][0]["message"].get("content") or "")[:80]
    print(f"warmup {dt:.1f}s {text!r} usage={j.get('usage')}", flush=True)

    # cheap follow-up — should be sub-second after the fat prefill
    t1 = time.time()
    body2 = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: SPARK-TR3-OK"}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    req2 = urllib.request.Request(
        args.api.rstrip("/") + "/chat/completions",
        data=json.dumps(body2).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req2, timeout=60) as r:
        j2 = json.loads(r.read())
    print(f"followup {time.time()-t1:.1f}s {j2['choices'][0]['message'].get('content')!r}", flush=True)


if __name__ == "__main__":
    main()
