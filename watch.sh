#!/bin/bash
# Poll a kernel and the LB, tolerating transient network failures.
while true; do
  K=$(python3 -m kaggle kernels status eggsysy/biohub-cache-train 2>/dev/null | tail -1)
  L=$(python3 -m kaggle competitions submissions -c biohub-cell-tracking-during-development 2>/dev/null | sed -n '3p' | grep -oE 'SubmissionStatus\.[A-Z]+ +[0-9.]*')
  echo "$(date +%H:%M) kernel=${K:-<net error>} | lb=${L:-<net error>}"
  case "$K$L" in
    *COMPLETE*COMPLETE*) echo BOTH_DONE; break;;
    *ERROR*) echo KERNEL_ERROR; break;;
  esac
  sleep 300
done
