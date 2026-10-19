#!/bin/bash
while true; do
  STATUS=$(/Users/aryanbadmera/Library/Python/3.14/bin/kaggle kernels status eggsysy/biohub-948-sew20)
  echo "$(date): $STATUS"
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.COMPLETE"'; then
    echo "Kernel completed successfully! Submitting to competition..."
    /Users/aryanbadmera/Library/Python/3.14/bin/kaggle competitions submit -c biohub-cell-tracking-during-development -k eggsysy/biohub-948-sew20 -f submission.csv -m "Submitting 0.948 pre-trained ensemble"
    break
  fi
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.ERROR"'; then
    echo "Kernel failed!"
    break
  fi
  
  if echo "$STATUS" | grep -q 'has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'; then
    echo "Kernel was cancelled."
    break
  fi
  
  sleep 30
done
