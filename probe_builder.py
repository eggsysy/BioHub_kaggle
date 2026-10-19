"""Build a single-knob upstream probe from the 0.935 notebook.

Cell 5 is a configuration guard that raises if certain knobs drift from their
banked values, so any probe touching a guarded knob must update the guard's
expectation too -- otherwise the kernel dies before predicting anything.
"""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(name: str, slug: str, changes: dict[str, str], guard: dict[str, float], doc: str):
    nb = json.loads((ROOT / "best_ensemble_0.935.ipynb").read_text())

    # --- env overrides -----------------------------------------------------
    cfg = nb["cells"][4]
    src = "".join(cfg["source"])
    assert "BIOHUB_PRESET" in src
    existing = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Subscript) and ast.unparse(t.value) == "os.environ":
                existing[ast.literal_eval(t.slice)] = ast.literal_eval(node.value)
    for k, v in changes.items():
        print(f"  {k:42s} {existing.get(k, '<default>')} -> {v}")
        if existing.get(k) == v:
            sys.exit(f"no-op: {k} already {v}")
    cfg["source"] = src + f"\n# ---- {doc} ----\n" + \
        "\n".join(f'os.environ["{k}"] = "{v}"' for k, v in changes.items()) + "\n"

    # --- keep the guard consistent with the probe --------------------------
    g = nb["cells"][5]
    gsrc = "".join(g["source"])
    for k, want in guard.items():
        old = f'"{k}": '
        i = gsrc.index(old) + len(old)
        j = gsrc.index(",", i)
        print(f"  guard {k:36s} {gsrc[i:j]} -> {want}")
        gsrc = gsrc[:i] + str(want) + gsrc[j:]
    g["source"] = gsrc

    # Cell 8 embeds a second, independent guard on the bidirectional weight.
    if "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" in changes:
        want = changes["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"]
        c8 = nb["cells"][8]
        s8 = "".join(c8["source"])
        for old, new in (
            ("_bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12",
             f"_bidirectional_weight_guard, {want}, rel_tol=0.0, abs_tol=1e-12"),
            ('"expected_bidirectional_weight": 0.15,',
             f'"expected_bidirectional_weight": {want},'),
        ):
            if old not in s8:
                sys.exit(f"cell 8 guard pattern not found: {old!r}")
            s8 = s8.replace(old, new)
        c8["source"] = s8
        print(f"  cell8 guard  0.15 -> {want}")

    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    d = ROOT / f"probe_{name}"
    d.mkdir(exist_ok=True)
    (d / f"{name}.ipynb").write_text(json.dumps(nb))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"eggsysy/{slug}", "title": slug, "code_file": f"{name}.ipynb",
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
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }, indent=2))
    print(f"  -> built probe_{name}/{name}.ipynb\n")


print("bidir 0.15 -> 0.30  (+0.00172 on 16 samples; gains via divisions)")
build("bidir030", "biohub-bidir030",
      {"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.30", "BIOHUB_VALIDATOR_ENABLE": "0"},
      {"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.30},
      "upstream probe: reverse-time fusion weight 0.15 -> 0.30")

print("det 0.965 -> 0.980  (+0.00083 on 16 samples; gains via node-count bonus)")
build("det098", "biohub-det098",
      {"BIOHUB_DET_THRESHOLD": "0.980", "BIOHUB_VALIDATOR_ENABLE": "0"},
      {"BIOHUB_DET_THRESHOLD": 0.98},
      "upstream probe: detection threshold 0.965 -> 0.980")
