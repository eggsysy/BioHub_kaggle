#!/bin/bash
cd /Users/aryanbadmera/Desktop/BioHub
while true; do
  A=$(python3 -m kaggle kernels status eggsysy/biohub-det098 2>/dev/null | tail -1)
  B=$(python3 -m kaggle kernels status eggsysy/biohub-ft-calib 2>/dev/null | tail -1)
  busy=0
  case "$A" in *RUNNING*|*QUEUED*) busy=$((busy+1));; esac
  case "$B" in *RUNNING*|*QUEUED*) busy=$((busy+1));; esac
  echo "$(date +%H:%M) busy=$busy  det098=$A  ftcalib=$B"
  if [ "$busy" -lt 2 ]; then
    python3 -m kaggle kernels push -p probe_bidir030 2>&1 | tail -1
    nohup ./submit_probe.sh biohub-bidir030 "Upstream probe: bidirectional fusion 0.15 -> 0.30 (+0.00172 offline)" > probe_bidir030.log 2>&1 &
    break
  fi
  sleep 180
done
