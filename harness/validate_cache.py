"""Validate the 50-sample cache before trusting it to drive a search.

Two questions:
  1. Does the baseline config produce a division_jaccard near the ~0.27 the
     leaderboard implies, rather than the ~0.00 the 4 public samples showed?
  2. How is the wall time split between post-processing and scoring? That
     decides whether Optuna is feasible at 50 samples or needs a subset.
"""
import sys, time
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

CACHE = Path("cache/pred_cache")
stems = replay.discover_stems(CACHE)
print(f"loading {len(stems)} samples ...", flush=True)
t0 = time.time()
samples = [replay.load_sample(s, CACHE) for s in stems]
n_dc = sum(1 for s in samples if s.dc_score)
print(f"loaded in {time.time()-t0:.0f}s; {n_dc}/{len(samples)} have DeepCenter scores\n", flush=True)

t0 = time.time()
r = replay.score_samples(samples, {})
dt = time.time() - t0

print(f"\nBASELINE (0.935 env) over {len(samples)} samples   [{dt:.0f}s]")
print(f"  score              = {r['score']:.5f}")
print(f"  adj_edge_jaccard   = {r['adj_edge_jaccard']:.5f}")
print(f"  division_jaccard   = {r['division_jaccard']:.4f}   "
      f"(tp/fp/fn = {r['division_tp']}/{r['division_fp']}/{r['division_fn']})")
print(f"  node_recall        = {r['node_recall']:.4f}")
print(f"  DeepCenter lookups = {r['dc_counters']}")
print(f"\n  per-config cost    = {dt:.0f}s  ->  100 Optuna trials = {dt*100/3600:.1f} h")

ps = sorted(r["per_sample"], key=lambda x: x["adj_edge_jaccard"])
print("\n  weakest 5 samples:")
for x in ps[:5]:
    print(f"    {x['stem']:20s} adj={x['adj_edge_jaccard']:.4f} edge_J={x['edge_jaccard']:.4f} "
          f"div(tp/fp/fn)=({x['division_tp']}/{x['division_fp']}/{x['division_fn']})")
print("  strongest 3:")
for x in ps[-3:]:
    print(f"    {x['stem']:20s} adj={x['adj_edge_jaccard']:.4f} edge_J={x['edge_jaccard']:.4f}")
