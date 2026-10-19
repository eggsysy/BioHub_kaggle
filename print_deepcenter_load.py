import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "DEEPCENTER_VETO_DETECTOR" in source and "def " not in source:
            print(f"--- Cell {i} snippet ---")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if "DEEPCENTER_VETO_DETECTOR" in line:
                    print('\n'.join(lines[max(0, j-5):min(len(lines), j+5)]))
