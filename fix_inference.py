import json

with open('kernel-metadata.json', 'r') as f:
    meta = json.load(f)

if "eggsysy/biohub-trained-weights-v1" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("eggsysy/biohub-trained-weights-v1")

with open('kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

with open('cell_tracking_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell that does the pip install & shutil.copytree
# and append a command to copy the new weights
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'shutil.copytree' in "".join(cell['source']):
        source = "".join(cell['source'])
        if "biohub-trained-weights-v1" not in source:
            source += "\n# OVERWRITE weak weights with our newly trained weights\n"
            source += "import shutil\n"
            source += "shutil.copy('/kaggle/input/biohub-trained-weights-v1/edge_predictor_best.pth', f'{REPO_DIR}/weights/{METHOD}/split_0/edge_predictor_best.pth')\n"
            cell['source'] = source

with open('cell_tracking_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

