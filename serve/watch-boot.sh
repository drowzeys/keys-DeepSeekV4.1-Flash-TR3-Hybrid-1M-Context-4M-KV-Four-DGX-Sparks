#!/usr/bin/env bash
# Waits until the API answers or any rank container exits. Prints progress every 60s.
t0=$(date +%s)
while :; do
  el=$(( $(date +%s) - t0 ))
  if curl -sf -m 5 http://10.100.10.1:8000/v1/models >/dev/null 2>&1; then echo "READY after ${el}s"; exit 0; fi
  for ip in 10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5; do
    st=$(ssh -o BatchMode=yes -o ConnectTimeout=8 keyspark@$ip "docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' dsv41-native 2>&1")
    case "$st" in running*) ;; *) echo "RANK CONTAINER on $ip: $st after ${el}s"; exit 1;; esac
  done
  if (( el % 60 < 30 )); then
    echo "[$el s] $(ssh -o BatchMode=yes keyspark@10.100.10.1 'docker logs --tail 1 dsv41-native 2>&1 | cut -c1-200; free -g | awk "/Mem:/{print \"avail=\"\$7\"G\"}"')"
  fi
  (( el > 5400 )) && { echo "TIMEOUT ${el}s"; exit 2; }
  sleep 30
done
