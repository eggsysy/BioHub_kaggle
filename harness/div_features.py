"""Phase 1: extract labelled division candidates for a learned classifier.

There is no learned division component anywhere in the deployed pipeline --
no ForkHead exists despite the notebook's markdown describing one. Divisions
come from the ILP permitting <=2 children plus hand-tuned geometric gates,
and score 18.7% precision / 19.2% recall. That is the gap this attacks.

For each sample we rebuild the graph with safe-divisions OFF, enumerate every
plausible fork (parent P, existing child C1, candidate second child Q), and
label it against the ground truth. Features are everything measurable without
GT at test time.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")

from pathlib import Path
import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import replay

SCALE = np.array(replay.VOXEL_SCALE_UM)
MATCH_UM = 7.0
CAND_RADIUS_UM = 16.0      # generous: the p99 of real sister separation is 17.1


def um(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * SCALE))


def load_gt(path: Path):
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"]).astype(int)
    pos = np.stack([np.asarray(g["nodes/props/%s/values" % k]).astype(float)
                    for k in "zyx"], axis=1)
    edges = np.asarray(g["edges/ids"])
    nodes = {int(i): (int(tt), p) for i, tt, p in zip(ids, t, pos)}
    children: dict[int, list[int]] = {}
    for s, d in edges:
        children.setdefault(int(s), []).append(int(d))
    return nodes, children


def match_to_gt(nodes_by_id, gt_nodes):
    """Predicted id -> GT id, optimal assignment per timepoint within 7um."""
    pred_t, gt_t = {}, {}
    for nid, n in nodes_by_id.items():
        pred_t.setdefault(int(n["t"]), []).append(nid)
    for gid, (tt, _) in gt_nodes.items():
        gt_t.setdefault(tt, []).append(gid)
    p2g = {}
    for t, pids in pred_t.items():
        gids = gt_t.get(t, [])
        if not gids:
            continue
        P = np.array([[nodes_by_id[i][k] for k in "zyx"] for i in pids]) * SCALE
        G = np.array([gt_nodes[g][1] for g in gids]) * SCALE
        cost = np.linalg.norm(P[:, None] - G[None], axis=-1)
        gated = np.where(cost <= MATCH_UM, cost, 1e6)
        ri, ci = linear_sum_assignment(gated)
        for r, c in zip(ri, ci):
            if gated[r, c] < 1e6:
                p2g[pids[r]] = gids[c]
    return p2g


def candidates_for(stem: str, cache: Path, ns):
    smp = replay.load_sample(stem, cache)
    counters = {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0}
    replay._install_cached_deepcenter(ns, smp, counters)
    nodes, edges, _ = ns["filter_output_graph"](
        dict(smp.nodes_by_id), [dict(e) for e in smp.raw_edges],
        dataset=stem, deepcenter_bundle={"cfg": None, "cached": True})

    gt_nodes, gt_children = load_gt(smp.gt_path)
    p2g = match_to_gt(nodes, gt_nodes)
    gt_div = {s: set(c) for s, c in gt_children.items() if len(c) >= 2}

    kids: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    prob: dict[tuple[int, int], float] = {}
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        kids.setdefault(s, []).append(d)
        parent[d] = s
        prob[(s, d)] = float(e.get("edge_prob") or 0.0)

    by_t: dict[int, list[int]] = {}
    for nid, n in nodes.items():
        by_t.setdefault(int(n["t"]), []).append(nid)
    trees = {}
    for t, ids in by_t.items():
        pts = np.array([[nodes[i][k] for k in "zyx"] for i in ids]) * SCALE
        trees[t] = (cKDTree(pts), ids)

    def pos(i):
        return [nodes[i][k] for k in "zyx"]

    def density(i, k=5):
        t = int(nodes[i]["t"])
        tree, ids = trees[t]
        d, _ = tree.query(np.array(pos(i)) * SCALE, k=min(k + 1, len(ids)))
        d = np.atleast_1d(d)
        return float(d[1:].mean()) if len(d) > 1 else 99.0

    rows = []
    for P, ch in kids.items():
        if len(ch) != 1:
            continue                      # already a fork, or a leaf
        C1 = ch[0]
        tP = int(nodes[P]["t"])
        tC = int(nodes[C1]["t"])
        if tC != tP + 1 or (tP + 1) not in trees:
            continue
        tree, ids = trees[tP + 1]
        near = tree.query_ball_point(np.array(pos(P)) * SCALE, CAND_RADIUS_UM)
        gP = p2g.get(P)
        for j in near:
            Q = ids[j]
            if Q == C1:
                continue
            qpar = parent.get(Q)
            if qpar == P:
                continue
            # label: does (P -> C1, Q) reproduce a GT division?
            y = 0
            if gP in gt_div:
                daughters = gt_div[gP]
                if {p2g.get(C1), p2g.get(Q)} == daughters:
                    y = 1
            d_PC1, d_PQ, d_C1Q = um(pos(P), pos(C1)), um(pos(P), pos(Q)), um(pos(C1), pos(Q))
            gc1 = kids.get(C1, [])
            gcq = kids.get(Q, [])
            diverge = (um(pos(gc1[0]), pos(gcq[0])) - d_C1Q) if (gc1 and gcq) else np.nan
            # how well is Q already explained by its current parent?
            q_par_d = um(pos(qpar), pos(Q)) if qpar is not None else np.nan
            q_par_p = prob.get((qpar, Q), np.nan) if qpar is not None else np.nan
            q_par_kids = len(kids.get(qpar, [])) if qpar is not None else 0
            rows.append(dict(
                stem=stem, P=P, C1=C1, Q=Q, y=y,
                d_parent_child=d_PC1, d_parent_cand=d_PQ, d_sister=d_C1Q,
                diverge=diverge,
                prob_PC1=prob.get((P, C1), np.nan),
                prob_PQ=prob.get((P, Q), np.nan),
                q_parent_dist=q_par_d, q_parent_prob=q_par_p,
                q_parent_nkids=q_par_kids,
                q_has_parent=0 if qpar is None else 1,
                dens_P=density(P), dens_C1=density(C1), dens_Q=density(Q),
                dc_P=smp.dc_score.get(P, np.nan),
                dc_C1=smp.dc_score.get(C1, np.nan),
                dc_Q=smp.dc_score.get(Q, np.nan),
                n_kids_C1=len(gc1), n_kids_Q=len(gcq),
                t_frac=tP / max(max(by_t), 1),
                frame_nodes=len(ids),
            ))
    return rows, len(gt_div)


if __name__ == "__main__":
    import csv, time
    CACHE = Path("cache/pred_cache")
    ns = replay.build_pipeline({"BIOHUB_OUTPUT_SAFE_DIVISIONS": "0"})
    stems = replay.discover_stems(CACHE)
    all_rows, tot_div = [], 0
    t0 = time.time()
    for i, s in enumerate(stems):
        r, nd = candidates_for(s, CACHE, ns)
        all_rows += r
        tot_div += nd
        print(f"[{i+1}/{len(stems)}] {s}: {len(r):6,} candidates, "
              f"{sum(x['y'] for x in r)} positive, {nd} GT divisions", flush=True)
    pos = sum(x["y"] for x in all_rows)
    print(f"\n{len(all_rows):,} candidates, {pos} positive ({pos/len(all_rows):.4%}), "
          f"{tot_div} GT divisions, recoverable={pos}/{tot_div} "
          f"[{time.time()-t0:.0f}s]")
    with open("div_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    print("wrote div_candidates.csv")
