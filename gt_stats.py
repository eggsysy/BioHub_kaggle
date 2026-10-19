"""Inventory the ground-truth graphs: annotation density and division counts."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

GT_DIR = Path(__file__).parent / "gt"


def read_geff(path: Path):
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"])
    edges = np.asarray(g["edges/ids"])
    return ids, t, edges


def estimated_nodes(path: Path):
    meta = json.loads((path / "zarr.json").read_text())

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

    return find(meta)


rows = []
for path in sorted(GT_DIR.glob("*.geff")):
    ids, t, edges = read_geff(path)
    out_deg = Counter(int(s) for s, _ in edges)
    divisions = sum(1 for v in out_deg.values() if v >= 2)
    # a "track" = weakly connected chain; count roots (no incoming edge)
    has_in = {int(d) for _, d in edges}
    roots = sum(1 for i in ids if int(i) not in has_in)
    stem = path.name.replace(".geff", "")
    rows.append({
        "stem": stem,
        "embryo": stem.split("_")[0],
        "gt_nodes": len(ids),
        "gt_edges": len(edges),
        "gt_divisions": divisions,
        "gt_tracks": roots,
        "n_frames": int(t.max()) + 1 if len(t) else 0,
        "est_total_nodes": estimated_nodes(path),
    })

with open(GT_DIR.parent / "gt_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

tot = lambda k: sum(r[k] for r in rows)
print(f"samples={len(rows)}  gt_nodes={tot('gt_nodes'):,}  gt_edges={tot('gt_edges'):,}  "
      f"gt_divisions={tot('gt_divisions'):,}  gt_tracks={tot('gt_tracks'):,}")
print(f"estimated true nodes across train = {tot('est_total_nodes'):,}")
print(f"annotation density = {tot('gt_nodes')/tot('est_total_nodes'):.3%} of real cells\n")

by_emb = defaultdict(lambda: Counter())
for r in rows:
    for k in ("gt_nodes", "gt_edges", "gt_divisions", "est_total_nodes"):
        by_emb[r["embryo"]][k] += r[k]
    by_emb[r["embryo"]]["n"] += 1
for emb, c in sorted(by_emb.items()):
    print(f"{emb}: n={c['n']:3d}  gt_nodes={c['gt_nodes']:6,}  gt_edges={c['gt_edges']:6,}  "
          f"gt_divisions={c['gt_divisions']:4,}  est_nodes={c['est_total_nodes']:9,}")

d = np.array([r["gt_divisions"] for r in rows])
print(f"\ndivisions/sample: mean={d.mean():.2f} median={np.median(d):.0f} "
      f"max={d.max()} zero-division samples={int((d == 0).sum())}/{len(d)}")
