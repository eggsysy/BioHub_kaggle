"""How much score is sitting in the node-count term?

The metric multiplies edge Jaccard by (1 - 0.1*(T_pred - T_true)/T_true), and
crucially a predicted node that matches no GT node is NOT a false positive --
it is ignored entirely, while still counting toward T_pred. Only ~2.8% of real
cells are annotated, so the overwhelming majority of predicted nodes are pure
T_pred cost with no metric benefit.

This measures the ceiling with an ORACLE: drop every predicted node that fails
to match a GT node. Unachievable in practice (we cannot see GT at test time),
but it sizes the prize. If the ceiling is large, a learned node-keep classifier
is worth building; if small, this whole direction dies here.
"""
import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import replay

CACHE = Path("cache/pred_cache")
SCALE = np.array(replay.VOXEL_SCALE_UM)
MAXD = 7.0


def matched_pred_ids(nodes_by_id, gt_nodes):
    """Per-timepoint optimal assignment within 7um -- same rule as the metric."""
    pred_by_t, gt_by_t = {}, {}
    for nid, n in nodes_by_id.items():
        pred_by_t.setdefault(int(n["t"]), []).append(nid)
    for gid, g in gt_nodes.items():
        gt_by_t.setdefault(int(g["t"]), []).append(gid)

    keep = set()
    for t, pids in pred_by_t.items():
        gids = gt_by_t.get(t, [])
        if not gids:
            continue
        P = np.array([[nodes_by_id[i][k] for k in "zyx"] for i in pids]) * SCALE
        G = np.array([[gt_nodes[g][k] for k in "zyx"] for g in gids]) * SCALE
        cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=-1)
        gated = np.where(cost <= MAXD, cost, 1e6)
        ri, ci = linear_sum_assignment(gated)
        for r, c in zip(ri, ci):
            if gated[r, c] < 1e6:
                keep.add(pids[r])
    return keep


def gt_nodes_of(path):
    import zarr
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"])
    z = np.asarray(g["nodes/props/z/values"])
    y = np.asarray(g["nodes/props/y/values"])
    x = np.asarray(g["nodes/props/x/values"])
    return {int(i): {"t": int(a), "z": float(b), "y": float(c), "x": float(d)}
            for i, a, b, c, d in zip(ids, t, z, y, x)}


from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

ns = replay.build_pipeline({})
stems = replay.discover_stems(CACHE)
rows_base, rows_oracle = [], []

for stem in stems:
    smp = replay.load_sample(stem, CACHE)
    counters = {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0}
    replay._install_cached_deepcenter(ns, smp, counters)
    nodes, edges, _ = ns["filter_output_graph"](
        dict(smp.nodes_by_id), [dict(e) for e in smp.raw_edges],
        dataset=stem, deepcenter_bundle={"cfg": None, "cached": True})

    gt = replay._graph_rows(smp.gt_path)
    for label, rows, keep in (("base", rows_base, None),
                              ("oracle", rows_oracle, matched_pred_ids(nodes, gt_nodes_of(smp.gt_path)))):
        if keep is None:
            n2, e2 = nodes, edges
        else:
            n2 = {k: v for k, v in nodes.items() if k in keep}
            e2 = [e for e in edges
                  if e["source_id"] in keep and e["target_id"] in keep]
        pred = replay._to_td_graph(n2, e2)
        er = evaluate(pred, gt, scale=replay.VOXEL_SCALE_UM, max_distance=MAXD)
        rows.append(per_sample_metrics(er, smp.n_total, node_recall(pred, gt)))
    print(".", end="", flush=True)

print()
for label, rows in (("baseline", rows_base), ("ORACLE prune", rows_oracle)):
    s = summarise(rows)
    tp = sum(r["edge_tp"] for r in rows); fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows); n = sum(r["num_pred_nodes"] for r in rows)
    print(f"{label:14s} score={s['score']:.5f} adj_edge={s['adj_edge_jaccard']:.5f} "
          f"edge_J={s['edge_jaccard']:.4f} div_J={s['division_jaccard']:.4f} "
          f"nodes={n:,} (tp/fp/fn={tp:,}/{fp:,}/{fn:,})")
