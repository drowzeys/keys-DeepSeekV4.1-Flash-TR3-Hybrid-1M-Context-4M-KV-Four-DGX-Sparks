#!/usr/bin/env python3
"""Needle-in-haystack at a target token count. usage: needle.py BASE TOKENS [depth 0..1]"""
import json, os, random, sys, time, urllib.request
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
BASE, TOK = sys.argv[1], int(sys.argv[2]); DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
random.seed(TOK)
code = f"{random.randint(100000, 999999)}-{random.choice('ABCDEFGH')}{random.randint(10,99)}"
fill = ["The harbor town kept its ledgers in copper ink and traded salt for timber each spring.",
        "Rain moved across the valley in slow grey bands while the mill wheel turned without hurry.",
        "A cartographer once measured the coast by counting her steps between the lighthouses.",
        "The orchard rows were planted north to south so the afternoon light reached every trunk."]
# ~20 tokens per sentence; build ~TOK tokens
n = TOK // 19
sents = [fill[i % 4] + f" (entry {i})" for i in range(n)]
pos = int(len(sents) * DEPTH)
sents.insert(pos, f"IMPORTANT ARCHIVE NOTE: the vault access code is {code}. Remember it exactly.")
hay = " ".join(sents)
prompt = hay + "\n\nQuestion: What is the vault access code stated in the archive note above? Answer with the code only."
body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 24, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.monotonic(); ttft = None; text = ""; usage = None
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        ev = json.loads(line[5:])
        if ev.get("choices") and ev["choices"][0]["delta"].get("content"):
            if ttft is None: ttft = time.monotonic() - t0
            text += ev["choices"][0]["delta"]["content"]
        if ev.get("usage"): usage = ev["usage"]
print(json.dumps(dict(target_tokens=TOK, prompt_tokens=usage and usage["prompt_tokens"], depth=DEPTH, ttft_s=round(ttft or -1, 1),
      total_s=round(time.monotonic() - t0, 1), expected=code, answer=text.strip(), PASS=code in text)))
