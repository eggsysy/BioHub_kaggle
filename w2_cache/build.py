"""Derive a cache-generating kernel from the 0.935 ensemble notebook.

Same pipeline, but the held-out validator is widened from 4 to N samples and
every raw prediction graph (plus its DeepCenter scores) is exported so that
post-processing can be replayed and tuned locally on CPU.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
N_PER_TYPE = 25          # 25 per embryo prefix x 2 embryos = 50 held-out samples

nb = json.loads((ROOT / "best_ensemble_0.935.ipynb").read_text())

# --- 1. widen the validator ------------------------------------------------
cfg = nb["cells"][4]
assert cfg["cell_type"] == "code" and "BIOHUB_PRESET" in "".join(cfg["source"])
cfg["source"] = "".join(cfg["source"]) + f'''

# ---- cache-run overrides (does not touch any guarded knob) ----
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "{N_PER_TYPE}"
'''

# --- 2. export the raw prediction graphs + DeepCenter scores ---------------
EXPORT = r'''
# ============================================================
# CACHE EXPORT -- raw prediction graphs for offline tuning
# ============================================================
# The .geff files under REPO_DIR/predictions are the pipeline output BEFORE
# filter_output_graph(). Exporting them lets every post-processing knob be
# retuned locally against the official metric without touching a GPU.
import tarfile
import numpy as np

CACHE_DIR = WORKING_DIR / "pred_cache"
CACHE_DIR.mkdir(exist_ok=True)

exported = []
for stem in val_stems:
    src = next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)
    if src is None:
        print("MISSING prediction for", stem)
        continue
    shutil.copytree(src, CACHE_DIR / f"{stem}.geff", dirs_exist_ok=True)
    exported.append(stem)
print(f"exported {len(exported)}/{len(val_stems)} prediction graphs")
'''

DC_EXPORT = r'''
# DeepCenter score for every predicted node, so the veto can be replayed
# offline. One heatmap pass per frame; nodes are scored in bulk per frame.
bundle = globals().get("DEEPCENTER_VETO_DETECTOR")
print("DeepCenter bundle loaded:", bundle is not None)

if bundle is not None:
    _real_test_dir = TEST_DIR
    globals()["TEST_DIR"] = TRAIN_DIR
    try:
        for stem in exported:
            pred_graph = graph_from_geff(CACHE_DIR / f"{stem}.geff")
            rows = list(pred_graph.node_attrs().iter_rows(named=True))
            by_t = {}
            for r in rows:
                by_t.setdefault(int(r["t"]), []).append(r)

            ids, scores = [], []
            for t in sorted(by_t):
                # Fresh caches per frame: one heatmap is computed, all nodes in
                # that frame are scored against it, then the memory is released.
                frame_cache, heatmap_cache = {}, {}
                for r in by_t[t]:
                    ids.append(int(r["node_id"]))
                    try:
                        s = deepcenter_score_point(
                            stem, int(r["t"]),
                            (float(r["z"]), float(r["y"]), float(r["x"])),
                            bundle, frame_cache, heatmap_cache,
                        )
                    except Exception:
                        s = float("nan")
                    scores.append(float("nan") if s is None else float(s))
            np.savez_compressed(
                CACHE_DIR / f"{stem}.dcscores.npz",
                node_id=np.asarray(ids, dtype=np.int64),
                score=np.asarray(scores, dtype=np.float32),
            )
            print(f"  {stem}: scored {len(ids)} nodes")
    finally:
        globals()["TEST_DIR"] = _real_test_dir
'''

TARBALL = r'''
# Ship the ground truth for the same stems too, so the offline harness is
# self-contained, then pack everything into one downloadable archive.
for stem in exported:
    gt = TRAIN_DIR / f"{stem}.geff"
    if gt.exists():
        shutil.copytree(gt, CACHE_DIR / f"{stem}.gt.geff", dirs_exist_ok=True)

(CACHE_DIR / "manifest.json").write_text(json.dumps({
    "stems": exported,
    "experiment_tag": EXPERIMENT_TAG,
    "det_threshold": DET_THRESHOLD,
    "note": "prediction graphs are pre-filter_output_graph",
}, indent=2))

out = WORKING_DIR / "pred_cache.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    tar.add(CACHE_DIR, arcname="pred_cache")
print("wrote", out, round(out.stat().st_size / 1e6, 1), "MB")
shutil.rmtree(CACHE_DIR)
'''

for src in (EXPORT, DC_EXPORT, TARBALL):
    nb["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    })

nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}

(OUT / "cache_train.ipynb").write_text(json.dumps(nb))
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "eggsysy/biohub-cache-train",
    "title": "biohub-cache-train",
    "code_file": "cache_train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
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
print(f"built cache_train.ipynb: {len(nb['cells'])} cells, N_PER_TYPE={N_PER_TYPE}")
