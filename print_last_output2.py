import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_output4/biohub-optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"Cell {i} execution_count: {cell.get('execution_count')}")
