"""Are the edge FNs caused by missed DETECTIONS or by wrong ASSOCIATIONS?

I concluded earlier that association is the bottleneck because node_recall is
0.98. But 0.98 recall over 30,594 GT nodes still leaves ~600 GT nodes with no
matching detection, and each of those can destroy up to two GT edges. If most
FNs trace to those misses, the lever is detection, not linking -- the opposite
of what I said.
"""
import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import numpy as np, zarr
from scipy.optimize import linear_sum_assignment
import replay

SCALE = np.array(replay.VOXEL_SCALE_UM); MAXD = 7.0
CACHE = Path("cache/pred_cache")


def gt_of(path):
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"]); t = np.asarray(g["nodes/props/t/values"]).astype(int)
    pos = np.stack([np.asarray(g[f"nodes/props/{k}/values"]).astype(float) for k in "zyx"], 1)
    e = np.asarray(g["edges/ids"])
    return {int(i): (int(a), p) for i, a, p in zip(ids, t, pos)}, [(int(s), int(d)) for s, d in e]


ns = replay.build_pipeline({})
tot = dict(fn=0, fn_src_miss=0, fn_tgt_miss=0, fn_both_matched=0, gt_nodes=0, gt_miss=0)
for stem in replay.discover_stems(CACHE):
    smp = replay.load_sample(stem, CACHE)
    replay._install_cached_deepcenter(ns, smp, {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0})
    nodes, edges, _ = ns["filter_output_graph"](
        dict(smp.nodes_by_id), [dict(e) for e in smp.raw_edges],
        dataset=stem, deepcenter_bundle={"cfg": None, "cached": True})
    gt_nodes, gt_edges = gt_of(smp.gt_path)

    pred_t, gt_t = {}, {}
    for nid, n in nodes.items():
        pred_t.setdefault(int(n["t"]), []).append(nid)
    for gid, (tt, _) in gt_nodes.items():
        gt_t.setdefault(tt, []).append(gid)
    g2p = {}
    for t, gids in gt_t.items():
        pids = pred_t.get(t, [])
        if not pids:
            continue
        G = np.array([gt_nodes[g][1] for g in gids]) * SCALE
        P = np.array([[nodes[i][k] for k in "zyx"] for i in pids]) * SCALE
        cost = np.linalg.norm(G[:, None] - P[None], axis=-1)
        gated = np.where(cost <= MAXD, cost, 1e6)
        ri, ci = linear_sum_assignment(gated)
        for r, c in zip(ri, ci):
            if gated[r, c] < 1e6:
                g2p[gids[r]] = pids[c]

    pred_edges = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    tot["gt_nodes"] += len(gt_nodes)
    tot["gt_miss"] += sum(1 for g in gt_nodes if g not in g2p)
    for s, d in gt_edges:
        ps, pd = g2p.get(s), g2p.get(d)
        if ps is not None and pd is not None and (ps, pd) in pred_edges:
            continue                     # TP
        tot["fn"] += 1
        if ps is None and pd is None:
            tot["fn_src_miss"] += 1      # both endpoints missed
        elif ps is None or pd is None:
            tot["fn_tgt_miss"] += 1      # one endpoint missed
        else:
            tot["fn_both_matched"] += 1  # both detected, link wrong/absent

print(f"GT nodes {tot['gt_nodes']:,}, undetected {tot['gt_miss']:,} "
      f"({tot['gt_miss']/tot['gt_nodes']:.2%})\n")
fn = tot["fn"]
print(f"edge FN total                       {fn:,}")
print(f"  both endpoints undetected         {tot['fn_src_miss']:,}  ({tot['fn_src_miss']/fn:.1%})")
print(f"  one endpoint undetected           {tot['fn_tgt_miss']:,}  ({tot['fn_tgt_miss']/fn:.1%})")
print(f"  BOTH detected, link missing/wrong {tot['fn_both_matched']:,}  ({tot['fn_both_matched']/fn:.1%})")
det = tot["fn_src_miss"] + tot["fn_tgt_miss"]
print(f"\n-> {det/fn:.0%} of edge FNs are caused by DETECTION misses")
print(f"-> {tot['fn_both_matched']/fn:.0%} are caused by ASSOCIATION errors")
