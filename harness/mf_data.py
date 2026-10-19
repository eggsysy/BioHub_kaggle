"""Build association training data: detected nodes + sparse GT link labels.

Tests one hypothesis: does a multi-frame temporal window beat pairwise
association? The shipped model only ever sees (t, t+1), and its errors are
provably symmetric (FP 51% / FN 49%, precision 0.9415 vs recall 0.9440), which
means its operating point is optimal and only a different model can help.

Supervision matches the official trainer: GT is sparse, so loss is taken only
on rows/columns whose node matched a GT node. Everything else is masked.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import numpy as np, zarr
from scipy.optimize import linear_sum_assignment
import replay

SCALE = np.array(replay.VOXEL_SCALE_UM)
MATCH_UM = 7.0


def load_gt(path: Path):
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"]).astype(int)
    pos = np.stack([np.asarray(g[f"nodes/props/{k}/values"]).astype(float)
                    for k in "zyx"], axis=1)
    edges = np.asarray(g["edges/ids"])
    return {int(i): (int(a), p) for i, a, p in zip(ids, t, pos)}, \
           {(int(s), int(d)) for s, d in edges}


def build(stem: str, cache: Path):
    """Return per-frame arrays: positions, and GT match ids."""
    smp = replay.load_sample(stem, cache)
    nodes = smp.nodes_by_id
    gt_nodes, gt_edges = load_gt(smp.gt_path)

    by_t: dict[int, list[int]] = {}
    for nid, n in nodes.items():
        by_t.setdefault(int(n["t"]), []).append(nid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (tt, _) in gt_nodes.items():
        gt_by_t.setdefault(tt, []).append(gid)

    frames = {}
    for t in sorted(by_t):
        pids = sorted(by_t[t])
        P = np.array([[nodes[i][k] for k in "zyx"] for i in pids], dtype=np.float32)
        match = np.full(len(pids), -1, dtype=np.int64)
        gids = gt_by_t.get(t, [])
        if gids:
            G = np.array([gt_nodes[g][1] for g in gids])
            cost = np.linalg.norm(P[:, None] * SCALE - G[None] * SCALE, axis=-1)
            gated = np.where(cost <= MATCH_UM, cost, 1e6)
            ri, ci = linear_sum_assignment(gated)
            for r, c in zip(ri, ci):
                if gated[r, c] < 1e6:
                    match[r] = gids[c]
        frames[t] = {"ids": np.array(pids), "pos": P, "gt": match}
    return frames, gt_edges


if __name__ == "__main__":
    import pickle, time
    CACHE = Path("cache/pred_cache")
    stems = replay.discover_stems(CACHE)
    out = {}
    t0 = time.time()
    for i, s in enumerate(stems):
        out[s] = build(s, CACHE)
        f, e = out[s]
        nm = sum(int((v["gt"] >= 0).sum()) for v in f.values())
        nn = sum(len(v["ids"]) for v in f.values())
        print(f"[{i+1}/{len(stems)}] {s}: {len(f)} frames, {nn:,} nodes, "
              f"{nm:,} GT-matched ({nm/nn:.1%}), {len(e):,} GT edges", flush=True)
    with open("mf_data.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote mf_data.pkl [{time.time()-t0:.0f}s]")
