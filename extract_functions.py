import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def filter_output_graph" in source or "def score_sample" in source or "def aggregate_official" in source:
            print(f"--- CELL {i} ---")
            print(source[:200]) # just peek to know which cell it is
