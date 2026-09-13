#!/usr/bin/env python3
"""Fixed-prompt quality probe. usage: quality_ab.py TAG
Sends a fixed prompt set at temperature 0 (greedy) to http://10.100.10.1:8000,
saves qa-<TAG>.json = {id: {prompt, answer, ok}}, auto-scores objective items.
Run once per serving config (same TAG scheme), then compare-quality.py."""
import json, os, sys, re, urllib.request

BASE = os.environ.get("BASE", "http://10.100.10.1:8000")
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
TAG = sys.argv[1]

# (id, prompt, max_tokens, checker) — checker(text)->bool or None (open/manual)
def has(*subs): return lambda t: all(s.lower() in t.lower() for s in subs)
def num_eq(val, tol=0.05): return lambda t: bool(re.findall(r"-?\d+\.?\d*", t.replace(",",""))) and abs(float(re.findall(r"-?\d+\.?\d*", t.replace(",",""))[-1]) - val) <= tol

PROMPTS = [
 ("math_avg", "A train travels 360 km in 4 hours, then 150 km in 2.5 hours. What is its average speed in km/h for the whole journey? End your reply with just the number.", 400, num_eq(78.46, 0.1)),
 ("math_gsm", "A shop sells pens at 3 for $2. Maria buys 18 pens and pays with a $20 bill. How much change does she get? End with just the dollar amount.", 400, num_eq(8.0)),
 ("reason_ages", "Tom is twice as old as Sara was when Tom was as old as Sara is now. Sara is 30. How old is Tom? End with just the number.", 500, num_eq(40.0)),
 ("fact_element", "What is the chemical element with atomic number 74? Give its name and one-or-two letter symbol.", 120, has("tungsten", "W")),
 ("fact_capital", "What is the capital city of Australia?", 60, has("canberra")),
 ("fact_speed", "What is the speed of light in vacuum in meters per second, to three significant figures?", 120, has("2.998", "10") if False else (lambda t: "3.00" in t or "2.998" in t or "299,792,458" in t.replace(" ","") or "299792458" in t.replace(",",""))),
 ("code_palin", "Write a Python function is_palindrome(s) that returns True if s is a palindrome ignoring case and non-alphanumeric characters. Output only the code in a single block.", 400, has("def is_palindrome")),
 ("instr_format", "Reply with exactly three words naming primary colors of light, each capitalized, separated by single spaces, and nothing else.", 40, lambda t: len([w for w in re.sub(r"[^A-Za-z ]","",t).split()])==3),
 ("reason_logic", "All bloops are razzies. All razzies are lazzies. Some lazzies are mazzies. Can we conclude that all bloops are lazzies? Answer yes or no and one short reason.", 200, has("yes")),
 ("prose_autumn", "Write exactly two sentences describing an autumn morning in a quiet town. Vivid but not purple.", 200, None),
]

out = {}
for pid, prompt, mt, chk in PROMPTS:
    body = {"model": MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": mt, "temperature": 0, "seed": 0}
    req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=300); j = json.load(r)
        txt = j["choices"][0]["message"]["content"]
    except Exception as e:
        txt = f"<ERROR {e}>"
    ok = None
    if chk is not None:
        try: ok = bool(chk(txt))
        except Exception: ok = False
    out[pid] = {"prompt": prompt, "answer": txt, "ok": ok}
    mark = {True:"PASS", False:"FAIL", None:"open"}[ok]
    print(f"[{mark}] {pid}: {txt[:90].replace(chr(10),' ')}")

scored = [v["ok"] for v in out.values() if v["ok"] is not None]
print(f"OBJECTIVE {sum(1 for x in scored if x)}/{len(scored)} correct")
json.dump(out, open(f"qa-{TAG}.json","w"), indent=1)
print(f"saved qa-{TAG}.json")
