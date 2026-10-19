"""Submission: the division config validated across both training embryos.

Found by Optuna over 120 trials on the 50-sample cache, using the harness that
reproduced the leaderboard ordering (0.935 / 0.912 / 0.908) before being
trusted. Held the lead for ~50 consecutive trials.

Effect is entirely division precision -- the edge score is untouched:

    adj_edge_jaccard   0.89671 -> 0.89675   (unchanged)
    division FP             61 -> 31        (halved)
    division TP             14 -> 15
    division_jaccard    0.1045 -> 0.1442

Generalises to both embryos, which is the check that matters given train/test
are embryo-disjoint:

    44b6  +0.00125
    6bba  +0.00520
    all   +0.00402   -> ~+0.010 LB at the measured 2.5x cache->LB calibration

Using trial 71: once both were embryo-split checked it weakly dominates --
identical on 44b6 (+0.00125) and one FP better on 6bba (+0.00542 vs +0.00520).

    44b6  +0.00125    6bba  +0.00542    all  +0.00413
"""
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

CHANGES = {
    "BIOHUB_SAFE_DIV_MAX_UM": "9.250193043603588",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "19.486653287190908",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "15.236789836845439",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.001706745704996418",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.0022919771899392125",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "3.686568907789547",
    "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "1",
    "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "1",
    "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.18808779448112303",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "0",
    "BIOHUB_VALIDATOR_ENABLE": "0",
}

nb = json.loads((ROOT / "best_ensemble_0.935.ipynb").read_text())
cfg = nb["cells"][4]
src = "".join(cfg["source"])
assert "BIOHUB_PRESET" in src

existing = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        if isinstance(t, ast.Subscript) and ast.unparse(t.value) == "os.environ":
            existing[ast.literal_eval(t.slice)] = ast.literal_eval(node.value)

changed = 0
for k, v in CHANGES.items():
    if existing.get(k) != v:
        changed += 1
    print(f"  {k:45s} {existing.get(k, '<default>')} -> {v}")
if changed == 0:
    raise SystemExit("no-op: every value already matches the baseline")

cfg["source"] = src + "\n# ---- Optuna division config (trial 71, embryo-validated) ----\n" + \
    "\n".join(f'os.environ["{k}"] = "{v}"' for k, v in CHANGES.items()) + "\n"

nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
(OUT / "sub_division.ipynb").write_text(json.dumps(nb))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "eggsysy/biohub-division-tuned",
    "title": "biohub-division-tuned",
    "code_file": "sub_division.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ],
    "kernel_sources": [],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}, indent=2))
print(f"\nbuilt sub_division.ipynb ({changed} knobs changed)")
