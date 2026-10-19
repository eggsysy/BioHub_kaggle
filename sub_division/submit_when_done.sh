#!/bin/bash
K=eggsysy/biohub-division-tuned
MSG="Optuna division config (trial 71): +0.00413 cache, validated on both embryos (44b6 +0.00125, 6bba +0.00542)"
while true; do
  S=$(python3 -m kaggle kernels status "$K" 2>/dev/null | tail -1)
  echo "$(date +%H:%M) ${S:-<net err>}"
  case "$S" in
    *COMPLETE*)
      python3 -m kaggle competitions submit -c biohub-cell-tracking-during-development \
        -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -3
      break;;
    *ERROR*|*CANCEL*) echo "kernel failed -- not submitting"; break;;
  esac
  sleep 120
done
