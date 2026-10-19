import json, pathlib

CODE = [
r'''
import os, json, tarfile, time
from pathlib import Path

CANDS = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
]
COMP = next((p for p in CANDS if p.exists()), None)
print("COMP:", COMP)
TRAIN = COMP / "train"
geffs = sorted(TRAIN.glob("*.geff"))
print("n .geff samples:", len(geffs))
tot = 0
for g in geffs:
    for f in g.rglob("*"):
        if f.is_file():
            tot += f.stat().st_size
print("total .geff bytes:", tot, "=", round(tot/1e6, 1), "MB")
''',
r'''
# Pack every ground-truth graph into one downloadable tarball.
OUT = Path("/kaggle/working/gt_geff.tar.gz")
t0 = time.time()
with tarfile.open(OUT, "w:gz") as tar:
    for g in geffs:
        tar.add(g, arcname=g.name)
print("wrote", OUT, round(OUT.stat().st_size/1e6, 1), "MB in", round(time.time()-t0, 1), "s")
''',
r'''
# Sample inventory: name -> embryo id, plus the node-count estimate the metric uses.
rows = []
for g in geffs:
    meta_path = g / "zarr.json"
    est = None
    try:
        meta = json.loads(meta_path.read_text())
        # estimated_number_of_nodes lives in the geff metadata block
        def find(d):
            if isinstance(d, dict):
                if "estimated_number_of_nodes" in d:
                    return d["estimated_number_of_nodes"]
                for v in d.values():
                    r = find(v)
                    if r is not None:
                        return r
            elif isinstance(d, list):
                for v in d:
                    r = find(v)
                    if r is not None:
                        return r
            return None
        est = find(meta)
    except Exception as e:
        print("meta fail", g.name, e)
    stem = g.name.replace(".geff", "")
    rows.append({"stem": stem, "embryo": stem.split("_")[0], "estimated_number_of_nodes": est})

import csv
with open("/kaggle/working/gt_inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stem", "embryo", "estimated_number_of_nodes"])
    w.writeheader()
    w.writerows(rows)

from collections import Counter
print("embryos:", Counter(r["embryo"] for r in rows))
print("missing estimate:", sum(1 for r in rows if r["estimated_number_of_nodes"] is None))
for r in rows[:5]:
    print(r)
''',
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"kernelspec": {"language": "python", "name": "python3"}},
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": c.strip().splitlines(keepends=True)}
        for c in CODE
    ],
}
p = pathlib.Path(__file__).parent / "gt_export.ipynb"
p.write_text(json.dumps(nb, indent=1))
print("wrote", p)
