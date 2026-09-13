#!/usr/bin/env python3
"""Functional checks against the DSV4.1-Flash endpoint: text, vision, tools, DSpark metrics."""
import base64, io, json, re, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.100.10.1:8000"
import os
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return {"choices": [{"message": {"content": f"HTTP {e.code}: {e.read()[:200]!r}"}, "finish_reason": "error"}], "usage": {"prompt_tokens": -1}}, time.monotonic() - t0

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)

# 1. text: arithmetic + factual + instruction following
r, dt = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 64,
    "messages": [{"role": "user", "content": "What is 17*23? Reply with only the number."}]})
txt = r["choices"][0]["message"]["content"].strip()
check("text arithmetic", "391" in txt, f"-> {txt!r} usage={r['usage']} {dt:.1f}s")

r, dt = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 120,
    "messages": [{"role": "user", "content": "Name the capital of Australia and the year Apollo 11 landed on the Moon. One short sentence."}]})
txt = r["choices"][0]["message"]["content"]
check("text factual", "Canberra" in txt and "1969" in txt, f"-> {txt.strip()!r}")

# 2. vision: rendered text + shapes (OCR + shape/color recognition), sent as data URL
img = Image.new("RGB", (640, 320), "white"); d = ImageDraw.Draw(img)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
except Exception: font = ImageFont.load_default()
d.text((30, 30), "SPARK 7731", fill="black", font=font)
d.ellipse((60, 150, 220, 310), fill="red"); d.rectangle((300, 150, 600, 300), fill="blue")
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
r, dt = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 160,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Transcribe the text in this image exactly, then list each shape with its color."}]}]})
txt = r["choices"][0]["message"]["content"]
low = txt.lower()
check("vision OCR+shapes", "7731" in txt and "red" in low and "blue" in low and ("circle" in low or "oval" in low) and ("rectangle" in low or "square" in low or "bar" in low),
      f"-> {txt.strip()!r} prompt_tokens={r['usage']['prompt_tokens']} {dt:.1f}s")

# 3. tool calling
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
r, dt = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 200, "tools": tools,
    "messages": [{"role": "user", "content": "What's the weather in Tokyo right now in celsius? Use the tool."}]})
msg = r["choices"][0]["message"]; calls = msg.get("tool_calls") or []
args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check("tool call parsed", calls and calls[0]["function"]["name"] == "get_weather" and args.get("city", "").lower().startswith("tokyo"),
      f"-> finish={r['choices'][0]['finish_reason']} calls={calls} content={msg.get('content')!r}")

# 4. DSpark acceptance from metrics
m = get("/metrics")
def metric(name):
    vals = re.findall(rf"^{name}(?:_total)?(?:\{{[^}}]*\}})? ([0-9.e+]+)$", m, re.M)
    return sum(float(v) for v in vals)
drafts, accepted, draft_tokens = metric("vllm:spec_decode_num_drafts"), metric("vllm:spec_decode_num_accepted_tokens"), metric("vllm:spec_decode_num_draft_tokens")
per_pos = [l for l in m.splitlines() if "spec_decode_num_accepted_tokens_per_pos" in l and not l.startswith("#")]
check("dspark metrics present", drafts > 0, f"drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} accepted={accepted:.0f} accept_rate={accepted/max(draft_tokens,1):.3f} mean_accepted_per_draft={accepted/max(drafts,1):.2f}")
for l in per_pos[:6]: print("   ", l)
print("OVERALL", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
