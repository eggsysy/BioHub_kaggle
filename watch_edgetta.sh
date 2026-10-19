#!/bin/bash
cd /Users/aryanbadmera/Desktop/BioHub
while true; do
  S=$(python3 -m kaggle kernels status eggsysy/biohub-edgetta 2>/dev/null | tail -1)
  echo "$(date +%H:%M) ${S:-<net err>}"
  case "$S" in
    *COMPLETE*)
      rm -rf sub_edgetta/output; mkdir -p sub_edgetta/output
      python3 -m kaggle kernels output eggsysy/biohub-edgetta -p sub_edgetta/output --force >/dev/null 2>&1
      echo "downloaded:"; ls -la sub_edgetta/output/submission.csv 2>/dev/null
      break;;
    *ERROR*|*CANCEL*) echo "FAILED"; break;;
  esac
  sleep 180
done
