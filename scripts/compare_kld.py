#!/usr/bin/env python3
"""compare_kld.py native tr3 exl3 — mean KL(native||quant) in nats over teacher-forced positions.
Truncated KL over the union of each position's top-N tokens (standard approximation);
also reports top-1 argmax agreement per position (exact, no alignment artifact since text is fixed)."""
import json, sys, math
tags=sys.argv[1:]; ref=tags[0]
D={t:json.load(open(f"kld-{t}.json")) for t in tags}
def dist_to_p(d):  # {token: logprob} -> normalized prob dict over its top-N
    if not d: return {}
    m=max(d.values()); ex={k:math.exp(v-m) for k,v in d.items()}; z=sum(ex.values())
    return {k:v/z for k,v in ex.items()}
print(f"reference = {ref}\n")
print(f"{'model':<8} {'mean_KL(nats)':>13} {'top1_agree':>11} {'positions':>10}")
for t in tags:
    if t==ref:
        print(f"{t:<8} {0.0:>13.4f} {1.000:>11.3f} {'-':>10}"); continue
    kls=[]; agree=[]; npos=0
    for s_ref,s_q in zip(D[ref],D[t]):
        for dr,dq in zip(s_ref["dists"], s_q["dists"]):
            if not dr or not dq: continue
            pr=dist_to_p(dr); pq=dist_to_p(dq)
            # truncated KL over ref's support; floor q to its smallest prob for missing tokens
            qfloor=min(pq.values())*0.1 if pq else 1e-6
            kl=sum(p*math.log(p/pq.get(k,qfloor)) for k,p in pr.items() if p>0)
            kls.append(max(kl,0.0))
            a=max(dr,key=dr.get); b=max(dq,key=dq.get); agree.append(1.0 if a==b else 0.0); npos+=1
    mk=sum(kls)/len(kls) if kls else float('nan')
    ag=sum(agree)/len(agree) if agree else float('nan')
    print(f"{t:<8} {mk:>13.4f} {ag:>11.3f} {npos:>10}")
print("\nLower KL = closer to the shipped reference. top1_agree here is exact (fixed forced text).")
