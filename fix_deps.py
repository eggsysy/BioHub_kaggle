import json

with open('cell_tracking_notebook.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code' and 'subprocess.run' in ''.join(cell['source']):
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        
        # Replace the pip install command to include --no-deps
        source = source.replace('"install", "--no-index"', '"install", "--no-deps", "--no-index"')
        
        cell['source'] = source

with open('cell_tracking_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

