import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][9]['source'])
if "datasets = " in source:
    print("datasets = IS IN CELL 9")
