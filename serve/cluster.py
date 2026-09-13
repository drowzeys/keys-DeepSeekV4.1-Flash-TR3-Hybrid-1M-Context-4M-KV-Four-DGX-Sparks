#!/usr/bin/env python3
"""Deploy and operate only this task's dsv41-native containers."""
import argparse
import concurrent.futures
import io
import pathlib
import shlex
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent
NODES = ['10.100.10.1', '10.100.10.2', '10.100.10.3', '10.100.10.5']
REMOTE = '/home/keyspark/dsv41-recovery-20260912'

def ssh(ip, command, data=None):
    p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                        '-o','ServerAliveInterval=30',f'keyspark@{ip}',command],
                       input=data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out=p.stdout.decode(errors='replace')
    print(f'[{ip}] exit={p.returncode}\n{out}', flush=True)
    return p.returncode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action', choices=['deploy','start','status','logs','stop'])
    ap.add_argument('--tail',type=int,default=30)
    a=ap.parse_args()
    if a.action=='deploy':
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as tar:
            for f in ['serve-rank.sh','patch','patch-extra','patch-upstream-boot10','patch-upstream-exl3-tp3e','engram_local.py','hybrid-port']:
                tar.add(ROOT/f,arcname=f,filter=lambda ti: None if '__pycache__' in ti.name else ti)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            codes=list(ex.map(lambda ip:ssh(ip,f'mkdir -p {REMOTE} && tar -xf - -C {REMOTE}',buf.getvalue()),NODES))
    elif a.action=='start':
        codes=[]
        for rank in [3,2,1,0]:
            import os
            env=' '.join(f'{k}={shlex.quote(os.environ[k])}' for k in ['GMU','MAXLEN','SEQS','BATCH','EAGER','SPEC','SPEC_K','TEXT_ONLY','ENGRAM_LOCAL','EXTRA','TR3','SERVED_NAME','TR3_DEBUG','TR3_NATIVE','TR3_NO_DECODE_PLAN','CUDAGRAPH_MODE','BREAKABLE','TR3_PREWARM','COMPILE_LEVEL','TR3_SKIP_IN_CAPTURE','KERNEL_CONFIG','IMAGE','PATCH_SET','CACHE_TAG','MODEL_DIR','TR3_FAST_MATH','TR3_DECODE_BLOCK_M','TR3_PREFILL_BLOCK_M','B12X_DYNAMIC_TILE_MN','ABLIT_CAPTURE','ABLIT_CAPTURE_TAG','ABLIT_HOST_DIR'] if k in os.environ)
            code=ssh(NODES[rank],f'{env} bash {REMOTE}/serve-rank.sh {rank}')
            codes.append(code)
            if code: break
    elif a.action=='stop':
        codes=[]
        for ip in NODES:
            command="if [ \"$(docker inspect --format '{{index .Config.Labels \"keyspark.task\"}}' dsv41-native 2>/dev/null)\" = dsv41-recovery-20260912 ]; then docker stop -t 25 dsv41-native && docker rm dsv41-native; fi"
            codes.append(ssh(ip,command))
    else:
        command=("docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}' dsv41-native; free -h; nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader" if a.action=='status' else f'docker logs --tail {a.tail} dsv41-native 2>&1')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            codes=list(ex.map(lambda ip:ssh(ip,command),NODES))
    raise SystemExit(int(any(codes)))

if __name__=='__main__': main()
