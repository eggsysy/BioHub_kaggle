import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "COMP_DIR" in source and "=" in source:
            for line in source.split('\n'):
                if "COMP_DIR" in line and "=" in line:
                    print(f"Cell {i}: {line.strip()}")
