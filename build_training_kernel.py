import json
import os

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.13",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "nvidiaTeslaT4",
      "dataSources": [
        {
          "sourceId": "69273",
          "sourceType": "competition",
          "datasetId": "69273"
        },
        {
          "sourceId": "pilkwang/biohub-tracking-support-pack-50ep-v1",
          "sourceType": "dataset",
          "datasetId": "5534289"
        }
      ],
      "isInternetEnabled": True,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": True
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# BioHub 3D UNet Fine-Tuning\n",
        "Fine-tunes the 50-epoch pre-trained weights for 10 epochs using a low learning rate."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install -q zarr rustworkx networkx numcodecs\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import shutil\n",
        "\n",
        "print(\"Extracting training repository...\")\n",
        "os.makedirs(\"/kaggle/working/repo\", exist_ok=True)\n",
        "!unzip -q -o /kaggle/input/biohub-tracking-support-pack-50ep-v1/tracking_repo.zip -d /kaggle/working/repo/\n",
        "!pip install -e /kaggle/working/repo\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"Patching training script to support full-model resuming...\")\n",
        "script_path = '/kaggle/working/repo/scripts/train_unet_transformer.py'\n",
        "with open(script_path, 'r') as f:\n",
        "    lines = f.readlines()\n",
        "\n",
        "for i, line in enumerate(lines):\n",
        "    if 'parser.add_argument(\"--unet-weights\"' in line:\n",
        "        lines.insert(i, '    parser.add_argument(\"--resume\", type=str, default=None, help=\"Full checkpoint to resume.\")\\n')\n",
        "        break\n",
        "for i, line in enumerate(lines):\n",
        "    if 'model = UNetNodeTransformer(' in line:\n",
        "        j = i\n",
        "        while ').to(device)' not in lines[j]: j += 1\n",
        "        lines.insert(j + 1, '''\n",
        "    if getattr(args, \"resume\", None) is not None:\n",
        "        state = torch.load(args.resume, map_location=\"cpu\", weights_only=True)\n",
        "        state = {k.replace(\"unet.module.\", \"unet.\", 1) if k.startswith(\"unet.\") else k: v for k, v in state.items()}\n",
        "        model.load_state_dict(state, strict=False)\n",
        "        print(f\"Resumed full model from {args.resume}\", flush=True)\n",
        "''')\n",
        "        break\n",
        "with open(script_path, 'w') as f:\n",
        "    f.writelines(lines)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python /kaggle/working/repo/scripts/train_unet_transformer.py \\\n",
        "    --data-dir /kaggle/input/biohub-cell-tracking-during-development/train \\\n",
        "    --resume /kaggle/input/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/checkpoint_last.pth \\\n",
        "    --epochs 10 \\\n",
        "    --lr 1e-5 \\\n",
        "    --batch-size 8 \\\n",
        "    --num-workers 2 \\\n",
        "    --det-loss-weight 1.0 \\\n",
        "    --det-neg-weight 0.01\n"
      ]
    }
  ]
}

os.makedirs("/Users/aryanbadmera/Desktop/BioHub/training_kernel", exist_ok=True)
with open("/Users/aryanbadmera/Desktop/BioHub/training_kernel/biohub-finetune.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

with open("/Users/aryanbadmera/Desktop/BioHub/training_kernel/kernel-metadata.json", "w") as f:
    json.dump({
      "id": "eggsysy/biohub-finetune",
      "title": "BioHub Fine-Tune UNet",
      "code_file": "biohub-finetune.ipynb",
      "language": "python",
      "kernel_type": "notebook",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
      "competition_sources": ["biohub-cell-tracking-during-development"],
      "kernel_sources": [],
      "model_sources": []
    }, f, indent=2)

print("Kaggle Kernel built.")
