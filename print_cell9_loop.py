import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][9]['source'])
lines = source.split('\n')
for i, line in enumerate(lines):
    if "DEEPCENTER_VETO_DETECTOR =" in line:
        print(f"Line {i}: {line}")
        print("Following lines:")
        print("\n".join(lines[i+1:i+15]))
        break
