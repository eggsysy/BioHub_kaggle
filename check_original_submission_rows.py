import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

source = "".join(nb['cells'][9]['source'])
if "submission_rows" in source:
    print(f"Cell 9 HAS submission_rows")
if "Inference Phase" in source:
    print(f"Cell 9 HAS Inference Phase")
