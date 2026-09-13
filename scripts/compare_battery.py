#!/usr/bin/env python3
"""compare_battery.py native tr3 exl3  — objective by category + per-item pass
grid + top-1 token agreement vs the reference on items where answers match."""
import json, sys
tags = sys.argv[1:]; ref = tags[0]
D = {t: json.load(open(f"bat-{t}.json")) for t in tags}
ids = list(D[ref].keys())

def top1_agree(a_top1, b_top1):
    n = min(len(a_top1), len(b_top1))
    if n == 0: return None
    return sum(1 for i in range(n) if a_top1[i]==b_top1[i]) / n

# per-category objective
print("== objective by category (correct/total)")
cats = {}
for t in tags:
    cc = {}
    for v in D[t].values():
        if v["ok"] is None: continue
        c=v["cat"]; cc.setdefault(c,[0,0]); cc[c][1]+=1; cc[c][0]+= 1 if v["ok"] else 0
    cats[t]=cc
allcats = sorted({c for t in tags for c in cats[t]})
print(f"{'cat':<9}"+" ".join(f"{t:>10}" for t in tags))
for c in allcats:
    print(f"{c:<9}"+" ".join(f"{cats[t].get(c,[0,0])[0]}/{cats[t].get(c,[0,0])[1]:>2}".rjust(10) for t in tags))
print(f"{'TOTAL':<9}"+" ".join(f"{sum(a for a,_ in cats[t].values())}/{sum(n for _,n in cats[t].values()):>2}".rjust(10) for t in tags))

# per-item disagreements (where quant differs from ref on objective)
print("\n== items where a quant disagrees with "+ref+" (objective)")
any_dis=False
for pid in ids:
    r=D[ref][pid]["ok"]
    row=[(t, D[t][pid]["ok"]) for t in tags]
    if any(ok!=r for t,ok in row if D[t][pid]["ok"] is not None):
        any_dis=True
        print(f"  {pid:<10} "+" ".join(f"{t}={ {True:'P',False:'F',None:'-'}[D[t][pid]['ok']] }" for t in tags))
if not any_dis: print("  (none — all configs agree on every objective item)")

# top-1 token agreement vs ref
print("\n== mean top-1 token agreement vs "+ref+" (all items)")
for t in tags[1:]:
    ags=[]
    for pid in ids:
        a=top1_agree(D[ref][pid].get("top1") or [], D[t][pid].get("top1") or [])
        if a is not None: ags.append(a)
    print(f"  {t}: {sum(ags)/len(ags):.3f}  (n={len(ags)})")

# mean generated-token logprob (confidence) per config
print("\n== mean generated-token logprob (higher=more confident)")
for t in tags:
    vs=[v["toklp"] for v in D[t].values() if v.get("toklp") is not None]
    print(f"  {t}: {sum(vs)/len(vs):.3f}")
