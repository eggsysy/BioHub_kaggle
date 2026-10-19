"""Score a submission.csv against ground truth with the official metric.

The four example test samples are copies of training samples, so their GT is
available locally. That makes any submission.csv directly checkable before a
submission slot is spent on it.

Usage:
    python harness/score_submission.py final_submission_output/submission.csv
    python harness/score_submission.py a.csv b.csv        # A/B two runs
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(ROOT / "official_repo" / "src"))
sys.path.insert(0, str(HARNESS))

import replay  # noqa: E402

GT_DIR = ROOT / "gt"


def read_submission(path: Path) -> dict[str, tuple[dict, list]]:
    """Return {dataset: (nodes_by_id, edges)} from a submission CSV."""
    nodes: dict[str, dict[int, dict]] = defaultdict(dict)
    edges: dict[str, list[dict]] = defaultdict(list)
    csv.field_size_limit(10**9)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if row["row_type"] == "node":
                nid = int(row["node_id"])
                nodes[ds][nid] = {
                    "node_id": nid, "t": int(row["t"]),
                    "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
                }
            else:
                edges[ds].append({
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                })
    return {ds: (nodes[ds], edges[ds]) for ds in nodes}


def score(path: Path, verbose: bool = True) -> dict:
    from tracking_cellmot.metrics import (
        evaluate, node_recall, per_sample_metrics, summarise,
    )

    data = read_submission(path)
    rows, per_sample = [], []
    for ds in sorted(data):
        gt_path = GT_DIR / f"{ds}.geff"
        if not gt_path.exists():
            if verbose:
                print(f"  {ds:20s} (no local GT -- skipped)")
            continue
        nodes_by_id, edges = data[ds]
        pred = replay._to_td_graph(nodes_by_id, edges)
        gt = replay._graph_rows(gt_path)
        er = evaluate(pred, gt, scale=replay.VOXEL_SCALE_UM, max_distance=7.0)
        row = per_sample_metrics(er, replay._estimated_nodes(gt_path), node_recall(pred, gt))
        rows.append(row)
        per_sample.append({"stem": ds, **row})
        if verbose:
            print(f"  {ds:20s} edge_J={row['edge_jaccard']:.4f} "
                  f"adj={row['adj_edge_jaccard']:.4f} "
                  f"div(tp/fp/fn)=({er.division_tp}/{er.division_fp}/{er.division_fn}) "
                  f"nodes={er.num_pred_nodes:,}")
    out = summarise(rows)
    out["per_sample"] = per_sample
    return out


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    results = {}
    for p in paths:
        print(f"\n=== {p}")
        results[p] = score(p)
        r = results[p]
        print(f"  {'TOTAL':20s} score={r['score']:.5f}  adj_edge={r['adj_edge_jaccard']:.5f}  "
              f"div_J={r['division_jaccard']:.4f} "
              f"(tp/fp/fn = {r['division_tp']}/{r['division_fp']}/{r['division_fn']})")
    if len(paths) == 2:
        a, b = results[paths[0]], results[paths[1]]
        print(f"\n=== delta ({paths[1].name} - {paths[0].name})")
        for k in ("score", "adj_edge_jaccard", "division_jaccard"):
            print(f"  {k:20s} {b[k] - a[k]:+.5f}   ({a[k]:.5f} -> {b[k]:.5f})")
        for k in ("division_tp", "division_fp", "division_fn"):
            print(f"  {k:20s} {b[k] - a[k]:+d}   ({a[k]} -> {b[k]})")


if __name__ == "__main__":
    main()
