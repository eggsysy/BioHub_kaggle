import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][9]['source'])
lines = source.split('\n')
for i, line in enumerate(lines):
    if "DEEPCENTER_VETO_DETECTOR =" in line:
        print("\n".join(lines[max(0, i-5):min(len(lines), i+15)]))
        break
