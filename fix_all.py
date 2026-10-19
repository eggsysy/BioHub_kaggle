import json

with open('cell_tracking_notebook.ipynb', 'r') as f:
    nb = json.load(f)

csv_patch = """import subprocess
import os
import pandas as pd

csv_script = \"\"\"
import os
from pathlib import Path
import pandas as pd
import tracksdata as td

REPO_DIR = "/kaggle/working/repo"
METHOD = "unet_transformer"
geffs = sorted(Path(REPO_DIR, "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
print(f"{len(geffs)} prediction graphs")

rows = []
for g in geffs:
    name = g.stem
    graph = td.graph.IndexedRXGraph.from_geff(g)
    graph = graph[0] if isinstance(graph, tuple) else graph
    for r in graph.node_attrs().iter_rows(named=True):
        rows.append({
            "dataset": name, "row_type": "node", "node_id": int(r["node_id"]),
            "t": int(r["t"]), "z": int(round(r["z"])), "y": int(round(r["y"])),
            "x": int(round(r["x"])), "source_id": -1, "target_id": -1,
        })
    for r in graph.edge_attrs().iter_rows(named=True):
        rows.append({
            "dataset": name, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
        })

submission = pd.DataFrame(
    rows,
    columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
)
submission.index.name = "id"
submission.to_csv("submission.csv")
print(f"Wrote submission.csv with {len(submission)} rows")
\"\"\"

with open("build_csv.py", "w") as f:
    f.write(csv_script)

env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/repo/src"
subprocess.run(["python", "build_csv.py"], env=env, check=True)

print(pd.read_csv("submission.csv").head())
"""

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Fix ARTIFACTS path
        if 'ARTIFACTS =' in source:
            source = source.replace('/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts/cellmot-baseline-artifacts', '/kaggle/input/cellmot-baseline-artifacts/cellmot-baseline-artifacts')
            cell['source'] = source
            
        # 2. Fix CSV cell
        elif 'submission.to_csv("submission.csv")' in source:
            cell['source'] = csv_patch

with open('cell_tracking_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

