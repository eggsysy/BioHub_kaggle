"""Replay the notebook's post-processing offline and score it with the official metric.

The 0.935 pipeline splits cleanly in two:

  GPU stage   detection -> edge scoring -> ILP  ->  a raw .geff per sample
  CPU stage   filter_output_graph(...)          ->  the submitted graph

Only the CPU stage is replayed here. It is pure graph work, so given a cached
raw .geff any of the ~40 post-processing knobs can be re-scored in seconds
instead of the ~3.2 GPU-minutes a full sample costs on Kaggle.

Two calls in that stage touch the raw image volume, which we do not have
locally:

  * the safe-division DeepCenter veto scores an EXISTING node, so the cached
    per-node score is exact;
  * the gap-closing veto scores a SYNTHETIC midpoint, which is served by a
    nearest-cached-node fallback and counted in ``dc_fallbacks``;
  * ``refine_synthetic_midpoint`` is image-only and stays disabled locally.

So division tuning is faithful; gap tuning is approximate. Confirm any winning
config on Kaggle before trusting its absolute number.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
CACHE_DIR = ROOT / "cache" / "pred_cache"

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

import sys
sys.path.insert(0, str(ROOT / "official_repo" / "src"))
from baseline_env import BASELINE_ENV  # noqa: E402

# Locally-forced settings: these paths need image data we do not have.
LOCAL_FORCED = {
    "BIOHUB_GAP_REFINE_SYNTHETIC": "0",
}


# ---------------------------------------------------------------- cache I/O
@dataclass
class SampleCache:
    stem: str
    nodes_by_id: dict[int, dict]
    raw_edges: list[dict]
    dc_score: dict[int, float]
    dc_tree: cKDTree | None
    dc_points: np.ndarray
    dc_ids: np.ndarray
    gt_path: Path
    n_total: float


def _graph_rows(geff_path: Path):
    import tracksdata as td
    g = td.graph.IndexedRXGraph.from_geff(geff_path)
    if isinstance(g, tuple):
        g = g[0]
    return g


def _estimated_nodes(geff_path: Path) -> float:
    meta = json.loads((geff_path / "zarr.json").read_text())

    def find(d):
        if isinstance(d, dict):
            if "estimated_number_of_nodes" in d:
                return d["estimated_number_of_nodes"]
            for v in d.values():
                if (r := find(v)) is not None:
                    return r
        elif isinstance(d, list):
            for v in d:
                if (r := find(v)) is not None:
                    return r
        return None

    v = find(meta)
    return float(v) if v is not None else float("nan")


def load_sample(stem: str, cache_dir: Path = CACHE_DIR) -> SampleCache:
    pred = _graph_rows(cache_dir / f"{stem}.geff")

    nodes_by_id: dict[int, dict] = {}
    for row in pred.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {
            "node_id": nid, "t": int(row["t"]),
            "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
        }
    raw_edges = []
    for row in pred.edge_attrs().iter_rows(named=True):
        prob = row.get("edge_prob")
        raw_edges.append({
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": None if prob is None else float(prob),
        })

    dc_score: dict[int, float] = {}
    dc_path = cache_dir / f"{stem}.dcscores.npz"
    if dc_path.exists():
        d = np.load(dc_path)
        dc_score = {int(i): float(s) for i, s in zip(d["node_id"], d["score"])}

    # Spatial index over scored nodes, for synthetic-point fallback.
    scored = [(nid, nodes_by_id[nid]) for nid in dc_score if nid in nodes_by_id]
    if scored:
        dc_ids = np.array([nid for nid, _ in scored], dtype=np.int64)
        dc_points = np.array(
            [[n["t"] * 1e4, n["z"], n["y"], n["x"]] for _, n in scored], dtype=np.float64,
        )
        dc_tree = cKDTree(dc_points)
    else:
        dc_ids, dc_points, dc_tree = np.empty(0, np.int64), np.empty((0, 4)), None

    return SampleCache(
        stem=stem, nodes_by_id=nodes_by_id, raw_edges=raw_edges,
        dc_score=dc_score, dc_tree=dc_tree, dc_points=dc_points, dc_ids=dc_ids,
        gt_path=cache_dir / f"{stem}.gt.geff",
        n_total=_estimated_nodes(cache_dir / f"{stem}.gt.geff"),
    )


# ------------------------------------------------------------ pipeline build
def build_pipeline(overrides: dict[str, str] | None = None) -> dict:
    """Exec the extracted config + post-processing into a fresh namespace.

    The config module reads os.environ at import time, so the environment is
    set first and the module source is re-executed per configuration.
    """
    env = dict(BASELINE_ENV)
    env.update(overrides or {})
    env.update(LOCAL_FORCED)

    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        ns: dict = {"__name__": "pipeline"}
        exec(compile((HARNESS / "pipeline_config.py").read_text(),
                     "pipeline_config.py", "exec"), ns)
        exec(compile((HARNESS / "pipeline_funcs.py").read_text(),
                     "pipeline_funcs.py", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ns["_env"] = env
    return ns


def _install_cached_deepcenter(ns: dict, sample: SampleCache, counters: dict) -> None:
    """Serve DeepCenter scores from the cache instead of the image volume."""
    nodes = sample.nodes_by_id
    by_coord = {}
    for nid, n in nodes.items():
        if nid in sample.dc_score:
            by_coord[(n["t"], round(n["z"], 3), round(n["y"], 3), round(n["x"], 3))] = nid

    def score_point(dataset, t, point, detector_bundle, frame_cache, heatmap_cache):
        if detector_bundle is None or not ns["USE_DEEPCENTER_VETO"]:
            return None
        key = (int(t), round(float(point[0]), 3),
               round(float(point[1]), 3), round(float(point[2]), 3))
        nid = by_coord.get(key)
        if nid is not None:
            counters["dc_exact"] += 1
            return sample.dc_score[nid]
        if sample.dc_tree is None:
            counters["dc_missing"] += 1
            return None
        # Synthetic point: fall back to the nearest scored node in the same frame.
        q = np.array([int(t) * 1e4, float(point[0]), float(point[1]), float(point[2])])
        dist, idx = sample.dc_tree.query(q, k=1)
        if dist >= 1e3:          # different frame -> no usable evidence
            counters["dc_missing"] += 1
            return None
        counters["dc_fallback"] += 1
        return sample.dc_score[int(sample.dc_ids[idx])]

    ns["deepcenter_score_point"] = score_point


# ------------------------------------------------------------------ scoring
def _to_td_graph(nodes_by_id: dict[int, dict], edges) -> "object":
    import polars as pl
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    ordered = sorted(nodes_by_id)
    remap = {}
    new_ids = graph.bulk_add_nodes([
        {"t": int(nodes_by_id[i]["t"]), "z": float(nodes_by_id[i]["z"]),
         "y": float(nodes_by_id[i]["y"]), "x": float(nodes_by_id[i]["x"])}
        for i in ordered
    ])
    remap = dict(zip(ordered, new_ids))

    payload = []
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        if s in remap and t in remap:
            payload.append({"source_id": remap[s], "target_id": remap[t]})
    if payload:
        graph.bulk_add_edges(payload)
    return graph


def score_samples(
    samples: list[SampleCache],
    overrides: dict[str, str] | None = None,
    verbose: bool = False,
) -> dict:
    """Run the post-processing at `overrides` and return the official summary."""
    from tracking_cellmot.metrics import (
        evaluate, node_recall, per_sample_metrics, summarise,
    )

    ns = build_pipeline(overrides)
    counters = {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0}
    rows, per_sample = [], []

    for sample in samples:
        _install_cached_deepcenter(ns, sample, counters)
        # A truthy bundle keeps the veto branches live; the patched scorer
        # never dereferences it.
        bundle = {"cfg": None, "cached": True}
        nodes, edges, stats = ns["filter_output_graph"](
            dict(sample.nodes_by_id),
            [dict(e) for e in sample.raw_edges],
            dataset=sample.stem,
            deepcenter_bundle=bundle,
        )
        pred_graph = _to_td_graph(nodes, edges)
        gt_graph = _graph_rows(sample.gt_path)

        er = evaluate(pred_graph, gt_graph, scale=VOXEL_SCALE_UM, max_distance=7.0)
        row = per_sample_metrics(er, sample.n_total, node_recall(pred_graph, gt_graph))
        rows.append(row)
        per_sample.append({"stem": sample.stem, **row})
        if verbose:
            print(f"  {sample.stem:24s} edge_J={row['edge_jaccard']:.4f} "
                  f"adj={row['adj_edge_jaccard']:.4f} "
                  f"div(tp/fp/fn)=({er.division_tp}/{er.division_fp}/{er.division_fn}) "
                  f"nodes={er.num_pred_nodes}")

    out = summarise(rows)
    out["per_sample"] = per_sample
    out["dc_counters"] = dict(counters)
    return out


def discover_stems(cache_dir: Path = CACHE_DIR) -> list[str]:
    manifest = cache_dir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["stems"]
    return sorted(
        p.name[:-5] for p in cache_dir.glob("*.geff") if not p.name.endswith(".gt.geff")
    )


# ------------------------------------------------------------- sanity guards
def assert_effective(overrides: dict[str, str]) -> None:
    """Fail loudly if an override cannot possibly change the offline result.

    The previous Optuna run sampled three parameters that never reached the
    pipeline and returned the same score 50 times. Every knob is checked here
    against two things: that the config module defines a constant for it, and
    that the post-processing module actually reads that constant. Upstream
    (prediction-stage) knobs such as the ILP weights or the detection
    threshold are baked into the cache and cannot be tuned offline.
    """
    cfg_src = (HARNESS / "pipeline_config.py").read_text()
    funcs_src = (HARNESS / "pipeline_funcs.py").read_text()

    dead, upstream = [], []
    for key in overrides:
        const = key[len("BIOHUB_"):] if key.startswith("BIOHUB_") else key
        if f"\n{const} = " not in cfg_src:
            dead.append(key)
        elif const not in funcs_src:
            upstream.append(key)

    problems = []
    if dead:
        problems.append(f"no config constant for: {sorted(dead)}")
    if upstream:
        problems.append(
            f"defined but never read by the post-processing stage, so it is "
            f"frozen in the cache and cannot be tuned offline: {sorted(upstream)}"
        )
    if problems:
        raise ValueError("ineffective overrides -- " + "; ".join(problems))


# --------------------------------------------------------------- parallelism
def _score_one(args):
    """Post-process and evaluate a single cached sample. Module-level so it can
    be sent to a worker process."""
    stem, cache_dir, overrides = args
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    sample = _WORKER_CACHE.get(stem)
    if sample is None:
        sample = load_sample(stem, cache_dir)
        _WORKER_CACHE[stem] = sample

    ns = _WORKER_NS.get(_key(overrides))
    if ns is None:
        ns = build_pipeline(overrides)
        _WORKER_NS.clear()          # one config in flight at a time
        _WORKER_NS[_key(overrides)] = ns

    counters = {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0}
    _install_cached_deepcenter(ns, sample, counters)
    nodes, edges, _ = ns["filter_output_graph"](
        dict(sample.nodes_by_id), [dict(e) for e in sample.raw_edges],
        dataset=sample.stem, deepcenter_bundle={"cfg": None, "cached": True},
    )
    pred = _to_td_graph(nodes, edges)
    gt = _graph_rows(sample.gt_path)
    er = evaluate(pred, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
    row = per_sample_metrics(er, sample.n_total, node_recall(pred, gt))
    return {"stem": stem, **row}, counters


_WORKER_CACHE: dict = {}
_WORKER_NS: dict = {}


def _key(overrides) -> str:
    return json.dumps(overrides or {}, sort_keys=True)


def score_stems_parallel(
    stems: list[str],
    overrides: dict[str, str] | None = None,
    cache_dir: Path = CACHE_DIR,
    workers: int | None = None,
) -> dict:
    """Same result as score_samples, spread over processes.

    Each worker keeps its loaded samples resident between calls, so repeated
    trials in a search pay the geff-parsing cost only once per worker.
    """
    import multiprocessing as mp
    from tracking_cellmot.metrics import summarise

    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        results = pool.map(_score_one, [(s, cache_dir, overrides) for s in stems])

    rows = [r for r, _ in results]
    counters = {"dc_exact": 0, "dc_fallback": 0, "dc_missing": 0}
    for _, c in results:
        for k in counters:
            counters[k] += c[k]
    out = summarise(rows)
    out["per_sample"] = rows
    out["dc_counters"] = counters
    return out
