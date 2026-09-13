#!/usr/bin/env bash
# Concurrency sweep; writes bench-<tag>.jsonl. usage: run-bench.sh TAG [max_tokens]
TAG=${1:?tag}; MT=${2:-256}; OUT=bench-$TAG.jsonl; : > $OUT
for c in 1 2 4 8; do python3 bench.py http://10.100.10.1:8000 $c $MT prose list code essay read math | tee -a $OUT; done
python3 - $OUT <<'PY'
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
cls=[]; [cls.append(r['cls']) for r in rows if r['cls'] not in cls]; concs=sorted({r['conc'] for r in rows})
def table(key,title):
    print(f"\n### {title}\n\n| class | "+" | ".join(f"C{c}" for c in concs)+" |\n|---|"+"---|"*len(concs))
    for k in cls: print(f"| {k} | "+" | ".join(next((str(r[key]) for r in rows if r['cls']==k and r['conc']==c),'-') for c in concs)+" |")
table('per_stream_decode_tps','Per-stream decode tok/s (after first token)'); table('aggregate_tps','Aggregate tok/s (wall, TTFT included)'); table('ttft_s','TTFT s')
PY
