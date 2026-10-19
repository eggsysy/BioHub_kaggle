import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "PIPELINE MANIFEST" in source:
            print("Skipping Pipeline Manifest")
            continue
    new_cells.append(cell)

nb['cells'] = new_cells

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'w') as f:
    json.dump(nb, f)
