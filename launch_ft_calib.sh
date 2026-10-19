#!/bin/bash
# Kaggle allows 2 concurrent GPU sessions. Wait for a probe to finish, then push.
cd /Users/aryanbadmera/Desktop/BioHub
while true; do
  A=$(python3 -m kaggle kernels status eggsysy/biohub-bidir030 2>/dev/null | tail -1)
  B=$(python3 -m kaggle kernels status eggsysy/biohub-det098 2>/dev/null | tail -1)
  busy=0
  case "$A" in *RUNNING*|*QUEUED*) busy=$((busy+1));; esac
  case "$B" in *RUNNING*|*QUEUED*) busy=$((busy+1));; esac
  echo "$(date +%H:%M) gpu sessions busy=$busy"
  if [ "$busy" -lt 2 ]; then
    echo "slot free -- pushing ft_calib"
    python3 -m kaggle kernels push -p ft_calib 2>&1 | tail -1
    break
  fi
  sleep 180
done
