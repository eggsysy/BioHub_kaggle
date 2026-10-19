import json

with open('/Users/aryanbadmera/Desktop/BioHub/best_ensemble_0.935.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 4 contains the configuration variables.
cell4_source = "".join(nb['cells'][4]['source'])

# Replace the specific lines
new_source = []
for line in cell4_source.split('\n'):
    if 'os.environ.get("BIOHUB_SAFE_DIV_MAX_UM"' in line:
        new_source.append('SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "12.485144"))')
    elif 'os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM"' in line:
        new_source.append('SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "8.766447"))')
    else:
        new_source.append(line)

# Since BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT is not explicitly in Cell 4 (it's pulled at runtime), 
# we can just add an explicit os.environ setter at the very top of Cell 4.
new_source.insert(0, 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.137349"\n')
new_source.insert(1, 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.485144"\n')
new_source.insert(2, 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.766447"\n')

nb['cells'][4]['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source]

with open('/Users/aryanbadmera/Desktop/BioHub/best_ensemble_optimized.ipynb', 'w') as f:
    json.dump(nb, f)

print("Created best_ensemble_optimized.ipynb")
