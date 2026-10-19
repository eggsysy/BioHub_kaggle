import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "TRAIN_DIR" in source and "=" in source:
            print(f"--- Cell {i} Snippet ---")
            for line in source.split('\n'):
                if "TRAIN_DIR" in line:
                    print(line)
