import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][9]['source'])
lines = source.split('\n')
print("\n".join(lines[-40:]))
