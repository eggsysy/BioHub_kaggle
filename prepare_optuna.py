import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "if VALIDATOR_ENABLE:" in source:
            print("FOUND VALIDATOR CELL:", i)
            with open('/Users/aryanbadmera/Desktop/BioHub/validator_cell.py', 'w') as out:
                out.write(source)
