#!/bin/bash
while true; do
  STATUS=$(/Users/aryanbadmera/Library/Python/3.14/bin/kaggle kernels status eggsysy/biohub-optuna-tuner)
  echo "$(date): $STATUS"
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.COMPLETE"'; then
    echo "Optuna completed successfully! Pulling output..."
    /Users/aryanbadmera/Library/Python/3.14/bin/kaggle kernels output eggsysy/biohub-optuna-tuner -p /Users/aryanbadmera/Desktop/BioHub/optuna_output --force
    cat /Users/aryanbadmera/Desktop/BioHub/optuna_output/biohub-optuna-tuner.log | grep "BEST "
    break
  fi
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.ERROR"'; then
    echo "Kernel failed!"
    break
  fi
  
  sleep 30
done
