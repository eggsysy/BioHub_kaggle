import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_output4/biohub-optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def filter_output_graph" in source:
            print(f"Cell {i} HAS filter_output_graph")
        if "def score_sample" in source:
            print(f"Cell {i} HAS score_sample")
