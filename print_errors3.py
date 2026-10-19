import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_output4/biohub-optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                print(f"Error in cell {i}:")
                print("".join(output.get('traceback', [])))
