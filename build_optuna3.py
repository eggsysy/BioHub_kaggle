import json

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Skip the submission generation cell
        if "submission_rows.extend(test_links)" in source or "df = pd.DataFrame(submission_rows" in source:
            continue
        if "if len(geffs) != len(test_stems):" in source or "test_stems = " in source:
            continue
        # Wait, if `test_stems` is not defined, we should just define it as empty so earlier cells don't crash
    new_cells.append(cell)

# Better: just explicitly define test_stems = [] right at the top!
new_cells.insert(1, {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["test_stems = []\n"]
})

nb['cells'] = new_cells

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'w') as f:
    json.dump(nb, f)
