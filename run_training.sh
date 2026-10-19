#!/bin/bash
set -e

echo "Setting up workspace..."
mkdir -p /workspace/data /workspace/weights /workspace/repo
cd /workspace

echo "Configuring Kaggle..."
export KAGGLE_API_TOKEN="KGAT_a44f2afe488e71075c2ee71ad60b09ac"

echo "Updating pip and installing dependencies..."
apt-get update && apt-get install -y unzip curl tmux
pip install -q --upgrade kaggle zarr rustworkx networkx numcodecs torch torchvision torchaudio monai

echo "Downloading competition dataset..."
kaggle competitions download -c biohub-cell-tracking-during-development -p data/ || { echo "Failed to download competition data"; exit 1; }
unzip -q -o data/biohub-cell-tracking-during-development.zip -d data/
rm data/biohub-cell-tracking-during-development.zip

echo "Downloading pre-trained weights..."
kaggle datasets download pilkwang/biohub-tracking-support-pack-50ep-v1 -p weights/ || { echo "Failed to download dataset"; exit 1; }
unzip -q -o weights/biohub-tracking-support-pack-50ep-v1.zip -d weights/
rm weights/biohub-tracking-support-pack-50ep-v1.zip

if [ -f "weights/tracking_repo.zip" ]; then
    unzip -q -o weights/tracking_repo.zip -d repo/
elif [ -d "weights/tracking_repo" ]; then
    cp -r weights/tracking_repo/* repo/
fi

echo "Patching training script..."
cat << 'PATCH_EOF' > patch_script.py
import os
script_path = '/workspace/repo/scripts/train_unet_transformer.py'
if os.path.exists(script_path):
    with open(script_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'parser.add_argument("--unet-weights"' in line:
            lines.insert(i, '    parser.add_argument("--resume", type=str, default=None, help="Full checkpoint to resume.")\n')
            break
    for i, line in enumerate(lines):
        if 'model = UNetNodeTransformer(' in line:
            j = i
            while ').to(device)' not in lines[j]: j += 1
            lines.insert(j + 1, '''
    if getattr(args, "resume", None) is not None:
        import torch
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        state = {k.replace("unet.module.", "unet.", 1) if k.startswith("unet.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Resumed full model from {args.resume}", flush=True)
''')
            break
    with open(script_path, 'w') as f:
        f.writelines(lines)
    print("Patched!")
PATCH_EOF
python3 patch_script.py

echo "Installing repo..."
pip install -e repo/

echo "Starting training..."
python3 repo/scripts/train_unet_transformer.py \
    --data-dir /workspace/data/train \
    --resume /workspace/weights/unet_transformer/split_0/checkpoint_last.pth \
    --epochs 10 \
    --lr 1e-5 \
    --batch-size 8 \
    --num-workers 8 > /workspace/training.log 2>&1

echo "DONE" > /workspace/status.txt
