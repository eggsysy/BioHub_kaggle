import json

with open('/Users/aryanbadmera/Desktop/BioHub/best_ensemble_optimized.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 4 currently starts with:
# os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.137349"
# os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.485144"
# os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.766447"
# import os

# Let's fix that by removing those first three lines if they exist, and injecting them after import os.
cell4 = nb['cells'][4]
source = "".join(cell4['source']).split('\n')

# Clean out the bad injected lines
clean_source = [line for line in source if not (line.startswith('os.environ["BIOHUB') and "=" in line and "12.485" in line or "8.766" in line or "0.137" in line)]

# Find 'import os'
import_idx = 0
for i, line in enumerate(clean_source):
    if "import os" in line:
        import_idx = i
        break

# Inject after import os
injections = [
    'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.137349"',
    'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "12.485144"',
    'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.766447"'
]

for idx, inj in enumerate(injections):
    clean_source.insert(import_idx + 1 + idx, inj)

cell4['source'] = [line + '\n' if not line.endswith('\n') else line for line in clean_source]

with open('/Users/aryanbadmera/Desktop/BioHub/best_ensemble_optimized.ipynb', 'w') as f:
    json.dump(nb, f)

print("Fixed best_ensemble_optimized.ipynb")
