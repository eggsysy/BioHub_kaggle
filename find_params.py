import json

with open('/Users/aryanbadmera/Desktop/BioHub/best_ensemble_0.935.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" in source or "BIOHUB_SAFE_DIV_MAX_UM" in source:
            print(f"--- Cell {i} ---")
            for line in source.split('\n'):
                if "BIOHUB_" in line:
                    print(line)
