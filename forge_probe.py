"""Single-knob probes on the 0.946 lineage-forge base.

Strategy that actually works here: one mechanically-motivated change per
submission, judged on the leaderboard. The offline proxy has mispredicted small
deltas repeatedly, but the LB has never lied, and we have ~5/day for 18 days.

Guard in cell 1 covers only DET_THRESHOLD, ILP_*, GAP_CLOSE_UM,
OUTPUT_MIN_TRACK_LEN and BIDIRECTIONAL_EDGE_WEIGHT, so safe-division knobs can
be changed without touching it.
"""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUARDED = {"BIOHUB_DET_THRESHOLD", "BIOHUB_ILP_APPEARANCE_WEIGHT",
           "BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "BIOHUB_GAP_CLOSE_UM",
           "BIOHUB_OUTPUT_MIN_TRACK_LEN", "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"}


def build(name, changes, note):
    nb = json.loads((ROOT / "sub_forge" / "forge.ipynb").read_text())
    cell = nb["cells"][0]
    src = "".join(cell["source"])
    env = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Subscript) and ast.unparse(t.value) == "os.environ":
                try: env[ast.literal_eval(t.slice)] = ast.literal_eval(n.value)
                except Exception: pass

    for k, v in changes.items():
        if k in GUARDED:
            sys.exit(f"{k} is guarded in cell 1 -- guard must be updated too")
        if env.get(k) == v:
            sys.exit(f"no-op: {k} already {v}")
        print(f"  {k:44s} {env.get(k, '<default>')} -> {v}")

    cell["source"] = src + f"\n# ---- probe: {note} ----\n" + \
        "\n".join(f'os.environ["{k}"] = "{v}"' for k, v in changes.items()) + "\n"
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    d = ROOT / f"probe_{name}"; d.mkdir(exist_ok=True)
    (d / f"{name}.ipynb").write_text(json.dumps(nb))

    built = "\n".join("".join(c["source"]) for c in json.loads((d / f"{name}.ipynb").read_text())["cells"])
    for k, v in changes.items():
        if f'os.environ["{k}"] = "{v}"' not in built:
            sys.exit(f"BUILD FAILED: {k} override missing from notebook")
    print("  verified: overrides present")

    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"eggsysy/biohub-{name}", "title": f"biohub-{name}",
        "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }, indent=2))
    print(f"  -> probe_{name}/\n")


if __name__ == "__main__":
    # Optuna (120 trials, old base) converged on sister 19.5; sjlee101 is
    # publicly testing 18.0; forge ships 14.0. Three signals, same direction.
    print("probe: sister_max 14 -> 18")
    build("sister18", {"BIOHUB_SAFE_DIV_SISTER_MAX_UM": "18.0"},
          "sister separation gate 14 -> 18 um (GT p95 is 15.3, p99 17.1)")

    # Optuna converged on 3.69 vs forge's 2.25. Median real divergence is 2.35,
    # so 2.25 admits nearly half of non-dividing pairs by coin flip.
    print("probe: diverge 2.25 -> 3.5")
    build("div35", {"BIOHUB_SAFE_DIV_DIVERGE_UM": "3.5"},
          "grandchild divergence gate 2.25 -> 3.5 um (stricter quality filter)")

    # Optuna's winning structure was: permissive geometry + TIGHT per-frame
    # budget. Forge has the permissive geometry (9.0/14.0) but still ships the
    # loose 0.0076 frame cap. This tests the other half of that structure.
    print("probe: frame cap 0.0076 -> 0.002")
    build("fcap002", {"BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.002"},
          "per-frame safe-division budget 0.0076 -> 0.002 (Optuna found 0.0017)")

    # Optuna converged on 15.24 against forge's 10.0.
    print("probe: existing-child 10 -> 15")
    build("child15", {"BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "15.0"},
          "existing-child distance gate 10 -> 15 um")

    # sjlee101 is publicly testing exactly this on the same base.
    print("probe: DC safe-div threshold 0.25 -> 0.20")
    build("dcthr020", {"BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20"},
          "DeepCenter safe-division veto threshold 0.25 -> 0.20 (looser veto)")
