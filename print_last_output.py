import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_output3/biohub-optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and len(cell.get('outputs', [])) > 0:
        print(f"Cell {i} has output.")
