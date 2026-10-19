import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_output4/biohub-optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Cell {i} length: {len(source)} bytes. Snippet: {source[:60].replace(chr(10), ' ')}")
