#!/bin/bash
while true; do
  STATUS=$(/Users/aryanbadmera/Library/Python/3.14/bin/kaggle kernels status eggsysy/biohub-optimized-ensemble)
  echo "$(date): $STATUS"
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.COMPLETE"'; then
    echo "Inference completed successfully! Downloading submission..."
    /Users/aryanbadmera/Library/Python/3.14/bin/kaggle kernels output eggsysy/biohub-optimized-ensemble -p /Users/aryanbadmera/Desktop/BioHub/final_submission_output --force
    
    echo "Submitting to Kaggle Competition..."
    /Users/aryanbadmera/Library/Python/3.14/bin/kaggle competitions submit -c biohub-cell-tracking-during-development -f /Users/aryanbadmera/Desktop/BioHub/final_submission_output/submission.csv -m "Optuna Optimized Pre-trained Ensemble (Target 0.96+)"
    
    break
  fi
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.ERROR"'; then
    echo "Kernel failed!"
    break
  fi
  
  sleep 60
done
