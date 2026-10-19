import json

with open('cell_tracking_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# The original pip install cell had this:
# subprocess.run(
#     ["pip", "install", "--no-deps", "--no-index", "--find-links", f"{ARTIFACTS}/wheels",
#      "tracksdata", "zarr>=3.0.10", "pyscipopt"],
#     check=True,
# )
# We want to replace it with the robust version.

robust_code = """import sys
import shutil
import subprocess
import os

WHEELS_DIR = "/kaggle/working/wheels"
shutil.copytree(f"{ARTIFACTS}/wheels", WHEELS_DIR, dirs_exist_ok=True)

for f in os.listdir(WHEELS_DIR):
    if f.startswith("numpy") or f.startswith("scipy"):
        os.remove(os.path.join(WHEELS_DIR, f))

subprocess.run(
    ["pip", "install", "--no-index", "--find-links", WHEELS_DIR,
     "tracksdata", "zarr>=3.0.10", "pyscipopt"],
    check=True,
)

shutil.copytree(f"{ARTIFACTS}/repo", REPO_DIR, dirs_exist_ok=True)
shutil.copytree(f"{ARTIFACTS}/weights", f"{REPO_DIR}/weights", dirs_exist_ok=True)
sys.path.insert(0, f"{REPO_DIR}/src")

print("Weights:", os.listdir(f"{REPO_DIR}/weights/{METHOD}/split_0"))
"""

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code' and 'subprocess.run' in ''.join(cell['source']) and 'pip' in ''.join(cell['source']):
        # Replace the entire cell source
        cell['source'] = robust_code

with open('cell_tracking_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

