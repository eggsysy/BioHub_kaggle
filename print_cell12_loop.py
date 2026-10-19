import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][12]['source'])
lines = source.split('\n')
for i, line in enumerate(lines):
    if "if VALIDATOR_ENABLE" in line:
        print(f"Line {i}: {line}")
        break
