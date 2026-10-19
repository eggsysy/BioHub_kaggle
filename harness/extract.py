"""Split the 0.935 notebook into an importable config module + pipeline module.

Cell 6 holds ~70 UPPER_CASE knobs read from the environment; cell 9 holds the
post-processing implementation (filter_output_graph and friends). Neither is
importable as-is, so we lift the constant assignments out of cell 6 with ast
and take cell 9 verbatim.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "best_ensemble_0.935.ipynb").read_text())

cfg_src = "".join(nb["cells"][6]["source"])
tree = ast.parse(cfg_src)

kept = []
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id.isupper():
            kept.append(ast.unparse(node))

(OUT / "pipeline_config.py").write_text(
    '"""Auto-generated from best_ensemble_0.935.ipynb cell 6 -- do not edit."""\n'
    "import os\n"
    "from pathlib import Path\n\n" + "\n".join(kept) + "\n"
)

# Cell 9 ends with the Kaggle submission driver (globs REPO_DIR, writes the
# CSV). Truncate after the last function definition so the module imports.
funcs_src = "".join(nb["cells"][9]["source"])
ftree = ast.parse(funcs_src)
last_def_end = max(
    n.end_lineno for n in ftree.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
)
funcs_src = "\n".join(funcs_src.splitlines()[:last_def_end])

# Cell 9 relies on names bound by earlier notebook cells. Supply them here.
# torch is deliberately None: the DeepCenter model can't run without image
# volumes, and every call site already treats a missing score as "accept".
PREAMBLE = """\
import math
import os
import json
import zarr

torch = None
nn = None
"""

(OUT / "pipeline_funcs.py").write_text(
    '"""Auto-generated from best_ensemble_0.935.ipynb cell 9 -- do not edit."""\n'
    + PREAMBLE + funcs_src + "\n"
)

# --- the exact env that produced the 0.935 LB score (cell 4) --------------
env_src = "".join(nb["cells"][4]["source"])
baseline = {}
for node in ast.parse(env_src).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        tgt = node.targets[0]
        if (isinstance(tgt, ast.Subscript)
                and ast.unparse(tgt.value) == "os.environ"):
            baseline[ast.literal_eval(tgt.slice)] = ast.literal_eval(node.value)

(OUT / "baseline_env.py").write_text(
    '"""The exact BIOHUB_* environment behind the 0.935 LB submission."""\n'
    "BASELINE_ENV = " + json.dumps(baseline, indent=4) + "\n"
)

print(f"pipeline_config.py: {len(kept)} constants")
print(f"baseline_env.py: {len(baseline)} env knobs")
print(f"pipeline_funcs.py: {len(funcs_src.splitlines())} lines (driver stripped)")
