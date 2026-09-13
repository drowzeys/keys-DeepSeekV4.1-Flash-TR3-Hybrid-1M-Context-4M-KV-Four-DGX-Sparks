#!/usr/bin/env python3
"""Teacher-forced per-token distribution capture for KLD. usage: kld.py TAG
Forces a fixed, diverse text set through the serve (echo + top_logprobs=20),
saves kld-<TAG>.json = [{tokens:[...], dists:[{tok:logprob,...}, ...]}] per sample.
Same text -> same token sequence across models, so compare_kld.py can KL them
position-by-position with no alignment artifact."""
import json, os, sys, urllib.request
BASE=os.environ.get("BASE","http://10.100.10.1:8000"); MODEL=os.environ.get("MODEL","deepseek-v4.1-flash"); TAG=sys.argv[1]
TOPN=20
SAMPLES=[
 "The 1854 Broad Street cholera outbreak in London was traced by the physician John Snow to a contaminated public water pump. By mapping the clustering of cases he identified the source and had the pump handle removed, which is now a founding example of epidemiology.",
 "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
 "To find the average speed over the whole journey, add the total distance and divide by the total time. 360 km plus 150 km is 510 km, and 4 hours plus 2.5 hours is 6.5 hours, so the average speed is about 78.5 km/h.",
 "The lighthouse keeper climbed the spiral stair each evening as the gulls wheeled over a slate-grey sea, and the lamp turned its slow patient circle against the coming dark.",
 "In a transformer, self-attention lets each token weigh every other token in the sequence; the query and key vectors produce scores, the softmax turns them into weights, and those weights mix the value vectors into the next representation.",
 "Question: A shop sells pens at three for two dollars. Maria buys eighteen pens and pays with a twenty-dollar bill. Answer: eighteen pens is six groups of three, costing twelve dollars, so her change from twenty dollars is eight dollars.",
]
out=[]
for text in SAMPLES:
    body={"model":MODEL,"prompt":text,"max_tokens":1,"echo":True,"logprobs":TOPN,"temperature":0}
    req=urllib.request.Request(BASE+"/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        r=urllib.request.urlopen(req,timeout=120); j=json.load(r); lp=j["choices"][0]["logprobs"]
        toks=lp["tokens"]; tops=lp["top_logprobs"]
        dists=[ (d or {}) for d in tops ]
        out.append({"tokens":toks,"dists":dists})
        print(f"  sample ok: {len(toks)} positions")
    except Exception as e:
        print(f"  sample ERROR {e}"); out.append({"tokens":[],"dists":[]})
json.dump(out, open(f"kld-{TAG}.json","w"))
print(f"saved kld-{TAG}.json ({len(out)} samples)")
