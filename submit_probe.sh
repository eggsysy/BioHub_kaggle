#!/bin/bash
# $1 = kernel slug, $2 = message
K="eggsysy/$1"
while true; do
  S=$(python3 -m kaggle kernels status "$K" 2>/dev/null | tail -1)
  case "$S" in
    *COMPLETE*)
      echo "$(date +%H:%M) $1 complete -- submitting"
      python3 -m kaggle competitions submit -c biohub-cell-tracking-during-development \
        -k "$K" -f submission.csv -v 1 -m "$2" 2>&1 | tail -2
      break;;
    *ERROR*|*CANCEL*) echo "$(date +%H:%M) $1 FAILED -- not submitting"; break;;
  esac
  sleep 120
done
