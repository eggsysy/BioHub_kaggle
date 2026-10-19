"""Single-knob probes on the 0.947 reyhan base (our current best)."""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUARDED = {"BIOHUB_DET_THRESHOLD", "BIOHUB_ILP_APPEARANCE_WEIGHT",
           "BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "BIOHUB_GAP_CLOSE_UM",
           "BIOHUB_OUTPUT_MIN_TRACK_LEN", "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"}


def build(name, changes, note):
    nb = json.loads((ROOT / "sub_reyhan" / "reyhan.ipynb").read_text())
    # Reyhan's notebook is one giant code cell that sets its env inline, so the
    # overrides must be spliced in directly AFTER the last os.environ line --
    # appending at the end would set them after the code has already read them.
    idx = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "os.environ[" in "".join(c["source"]))
    src = "".join(nb["cells"][idx]["source"])
    lines = src.splitlines(keepends=True)
    last_env = max(i for i, l in enumerate(lines) if l.lstrip().startswith("os.environ["))

    env = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Subscript) and ast.unparse(t.value) == "os.environ":
                try: env[ast.literal_eval(t.slice)] = ast.literal_eval(n.value)
                except Exception: pass

    for k, v in changes.items():
        if k in GUARDED: sys.exit(f"{k} is guarded")
        if env.get(k) == v: sys.exit(f"no-op: {k} already {v}")
        print(f"  {k:46s} {env.get(k,'<default>')} -> {v}")

    indent = lines[last_env][:len(lines[last_env]) - len(lines[last_env].lstrip())]
    inject = f"{indent}# ---- probe: {note} ----\n" + "".join(
        f'{indent}os.environ["{k}"] = "{v}"\n' for k, v in changes.items())
    new_src = "".join(lines[:last_env + 1]) + inject + "".join(lines[last_env + 1:])
    print(f"  spliced after line {last_env} of {len(lines)} (indent={len(indent)})")

    nb["cells"][idx]["source"] = new_src
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    d = ROOT / f"rp_{name}"; d.mkdir(exist_ok=True)
    (d / f"{name}.ipynb").write_text(json.dumps(nb))
    built = "\n".join("".join(c["source"]) for c in json.loads((d / f"{name}.ipynb").read_text())["cells"])
    for k, v in changes.items():
        if f'os.environ["{k}"] = "{v}"' not in built:
            sys.exit(f"BUILD FAILED: {k} missing")
    if "reyhanksatria" in built:
        sys.exit("BUILD FAILED: private dataset path leaked back in")
    print("  verified")
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
    print(f"  -> rp_{name}/\n")


if __name__ == "__main__":
    # mtoshidesu published exactly this delta against the same base.
    print("probe: secondary edge-TTA weight 0.75 -> 1.0")
    build("sectta10", {"BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "1.0"},
          "secondary edge-feature TTA weight 0.75 -> 1.0")

    # The secondary seed's contribution to edge scoring -- untouched by anyone
    # so far, and it sits directly on the association path where 55% of the
    # edge FNs live.
    print("probe: secondary edge weight 0.15 -> 0.22")
    build("sew022", {"BIOHUB_SECONDARY_EDGE_WEIGHT": "0.22"},
          "secondary model edge weight 0.15 -> 0.22")

    # SISTER_SYMMETRY_TAU is one of the mechanisms forge/reyhan added over the
    # old notebook and nobody has published a sweep of it.
    print("probe: sister symmetry tau 0.6 -> 0.45")
    build("tau045", {"BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.45"},
          "sister symmetry tau 0.6 -> 0.45 (looser symmetry requirement)")
