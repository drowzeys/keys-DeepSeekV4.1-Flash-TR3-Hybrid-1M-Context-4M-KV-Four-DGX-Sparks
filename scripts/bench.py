#!/usr/bin/env python3
"""Throughput benchmark: counts usage.completion_tokens (not SSE chunks), streams for TTFT.
usage: bench.py BASE CONCURRENCY [max_tokens] [classes...]"""
import asyncio, json, statistics, sys, time, aiohttp
BASE, CONC = sys.argv[1], int(sys.argv[2]); MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
import os
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
PROMPTS = {
 "prose": "Write a vivid 600-word short story about a lighthouse keeper who discovers the light has started responding to her thoughts.",
 "code": "Write a complete Python module implementing an LRU cache with TTL expiry, type hints, docstrings, and a small pytest test suite.",
 "list": "List 40 distinct programming languages with one sentence each on what they are best known for, as a numbered list.",
 "math": "Solve step by step: a train leaves at 3:15pm going 72 km/h; another leaves the same station at 4:00pm at 96 km/h on the same track. When and where does the second catch the first? Show all work, then give three similar problems with full solutions.",
 "essay": "Write a well-structured 900-word argumentative essay on whether cities should replace cars with public transit, with intro, three body paragraphs, and a conclusion.",
 "read": "Read this passage and answer in 2-3 sentences. Passage: The 1854 Broad Street cholera outbreak in London was traced by physician John Snow to a single contaminated water pump, which he identified by mapping cases; removing the pump handle helped end the outbreak and is a founding example of epidemiology. Question: How did John Snow identify the source of the outbreak and what action followed?",
 "json": "Produce a JSON array of 30 fictional employees with fields id, name, department, salary, start_date, skills (array of 3). Output only JSON.",
}
classes = sys.argv[4:] or list(PROMPTS)

async def one(session, cls, idx):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[cls] + f" (variant {idx})"}],
            "max_tokens": MAXTOK, "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); ttft = None; usage = None
    async with session.post(BASE + "/v1/chat/completions", json=body) as r:
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            ev = json.loads(line[5:])
            if ttft is None and ev.get("choices") and ev["choices"][0]["delta"].get("content"): ttft = time.monotonic() - t0
            if ev.get("usage"): usage = ev["usage"]
    t = time.monotonic() - t0
    return dict(cls=cls, ttft=ttft, total=t, out=usage["completion_tokens"], prompt=usage["prompt_tokens"], decode_tps=usage["completion_tokens"] / (t - (ttft or 0)))

async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        for cls in classes:
            t0 = time.monotonic()
            res = await asyncio.gather(*(one(s, cls, i) for i in range(CONC)))
            wall = time.monotonic() - t0
            tot = sum(r["out"] for r in res)
            print(json.dumps(dict(cls=cls, conc=CONC, per_stream_decode_tps=round(statistics.mean(r["decode_tps"] for r in res), 1),
                  aggregate_tps=round(tot / wall, 1), ttft_s=round(statistics.mean(r["ttft"] for r in res), 2),
                  out_tokens=tot, prompt_tokens=res[0]["prompt"], wall_s=round(wall, 1))), flush=True)
asyncio.run(main())
