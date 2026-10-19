"""Edge-feature TTA: reuse the 8 augmented forward passes already being computed.

The det_tta block runs model.encode() on 8 augmented views but discards the
UNet feature map every time -- `_, det_flip = model.encode(imgs_flip)`. Only
the detection logits are averaged. The edge scorer then indexes features from
the SINGLE unaugmented view.

So detection gets 8-view variance reduction and association gets none, despite
association accounting for 917 of 1,657 edge FNs (55%) plus most of the 1,734
FPs. Accumulating the feature maps costs one extra tensor of memory and zero
extra compute -- the forwards already happen.

Gated behind BIOHUB_EDGE_TTA so it is a single controlled change.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

nb = json.loads((ROOT / "best_ensemble_0.935.ipynb").read_text())

cfg = nb["cells"][4]
src = "".join(cfg["source"])
assert "BIOHUB_PRESET" in src
cfg["source"] = src + '\nos.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\nos.environ["BIOHUB_EDGE_TTA"] = "1"\n'

PATCH = r'''
# ============================================================
# EDGE-FEATURE TTA
# ============================================================
# Runs AFTER cell 8, so it targets the already-patched 8-view det_tta block.
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()

_old = """            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
_new = """            for f in range(W):
                det_logits[f] = det_logits[f] / _nv
            if os.environ.get("BIOHUB_EDGE_TTA", "0") != "0":
                unet_out = _uo_acc / _nv
            del _uo_acc"""
if _s.count(_old) != 1:
    raise RuntimeError(f"det_tta averaging anchor found {_s.count(_old)}x, expected 1")
_s = _s.replace(_old, _new, 1)

# accumulate the feature maps that are currently discarded
_edits = [
    ("""        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)""",
     """        if cfg.det_tta:
            _nv = 1
            _uo_acc = unet_out.clone()
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _uo, det_flip = model.encode(imgs_flip)"""),
    ("""                del imgs_flip, det_flip
                _nv += 1""",
     """                _uo_acc = _uo_acc + _uo.flip(dims)
                del imgs_flip, det_flip, _uo
                _nv += 1"""),
    ("""                _, det_rot = model.encode(imgs_rot)""",
     """                _uo, det_rot = model.encode(imgs_rot)"""),
    ("""                del imgs_rot, det_rot
                _nv += 1""",
     """                _uo_acc = _uo_acc + torch.rot90(_uo, -_k, dims=(-2, -1))
                del imgs_rot, det_rot, _uo
                _nv += 1"""),
    ("""            _, det_t = model.encode(imgs_t)""",
     """            _uo, det_t = model.encode(imgs_t)"""),
    ("""            del imgs_t, det_t
            _nv += 1""",
     """            _uo_acc = _uo_acc + _uo.transpose(-1, -2)
            del imgs_t, det_t, _uo
            _nv += 1"""),
    ("""            _, det_at = model.encode(imgs_at)""",
     """            _uo, det_at = model.encode(imgs_at)"""),
    ("""            del imgs_at, det_at
            _nv += 1""",
     """            _uo_acc = _uo_acc + torch.rot90(_uo.transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at, _uo
            _nv += 1"""),
]
for _o, _n in _edits:
    if _s.count(_o) != 1:
        raise RuntimeError(f"edge-TTA anchor found {_s.count(_o)}x, expected 1: {_o[:60]!r}")
    _s = _s.replace(_o, _n, 1)

_ps.write_text(_s)
print("edge-feature TTA patch applied: UNet features now averaged over all 8 TTA views")
print("  (previously discarded; detection was averaged, association was not)")
'''

nb["cells"].insert(9, {"cell_type": "code", "execution_count": None,
                       "metadata": {}, "outputs": [],
                       "source": PATCH.strip("\n").splitlines(keepends=True)})
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
(OUT / "sub_edgetta.ipynb").write_text(json.dumps(nb))

built = "\n".join("".join(c["source"]) for c in json.loads((OUT / "sub_edgetta.ipynb").read_text())["cells"])
for needle, why in [('os.environ["BIOHUB_EDGE_TTA"] = "1"', "env flag"),
                    ("_uo_acc = unet_out.clone()", "accumulator init"),
                    ("unet_out = _uo_acc / _nv", "feature averaging"),
                    ("edge-TTA anchor found", "anchor guards")]:
    if needle not in built:
        raise SystemExit(f"BUILD FAILED: {why} missing")
print("verified: all edits present")

(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "eggsysy/biohub-edgetta", "title": "biohub-edgetta",
    "code_file": "sub_edgetta.ipynb", "language": "python", "kernel_type": "notebook",
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
print(f"built sub_edgetta.ipynb ({len(nb['cells'])} cells)")
