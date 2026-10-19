with open("/Users/aryanbadmera/Desktop/BioHub/official_repo/scripts/train_unet_transformer.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'parser.add_argument("--unet-weights"' in line:
        lines.insert(i, '    parser.add_argument("--resume", type=str, default=None, help="Path to full model checkpoint to resume (UNet + Transformer).")\n')
        break

for i, line in enumerate(lines):
    if 'model = UNetNodeTransformer(' in line:
        # find the end of the instantiation
        j = i
        while ').to(device)' not in lines[j]:
            j += 1
        
        insert_code = """
    if getattr(args, "resume", None) is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        state = {k.replace("unet.module.", "unet.", 1) if k.startswith("unet.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"  Resumed full model from {args.resume}", flush=True)
"""
        lines.insert(j + 1, insert_code)
        break

with open("/Users/aryanbadmera/Desktop/BioHub/official_repo/scripts/train_unet_transformer.py", "w") as f:
    f.writelines(lines)
