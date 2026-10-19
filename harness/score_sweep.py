"""Score every upstream sweep config with the official metric.

Each config is a full cache of raw prediction graphs from one GPU pass. All
are post-processed identically, so differences are attributable to the single
upstream knob that changed. Config "base" repeats the shipped settings as a
matched control.

These caches carry no DeepCenter scores (re-scoring per config would have
doubled GPU time), so the DC veto behaves as accept-all throughout. That is a
constant across configs and should not disturb the ranking -- but any winner
needs a DC-enabled confirmation run before its absolute number is trusted.
"""
import sys, json, time
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

SWEEP = Path("cache/sweep_cache")
CONFIGS = ["base", "det0.940", "det0.980", "disapp1.0", "disapp3.5",
           "app0.5", "div0.8", "div1.8", "bidir0.00", "bidir0.30"]
WORKERS = 5   # the tracks search is using the other cores

# Score under the shipped post-processing, so this is comparable to the
# 0.935 baseline rather than entangled with the new division config.
POST = {}

stems = json.loads((SWEEP / "manifest.json").read_text())["stems"]
print(f"{len(stems)} samples per config, {len(CONFIGS)} configs\n")
print(f"{'config':12s} {'score':>9s} {'vs base':>9s} {'adj_edge':>9s} {'div_J':>7s} "
      f"{'nodes':>9s}  div(tp/fp/fn)")

base_score = None
rows = []
for tag in CONFIGS:
    d = SWEEP / tag
    if not d.exists():
        print(f"{tag:12s} MISSING")
        continue
    t0 = time.time()
    r = replay.score_stems_parallel(stems, POST, d, workers=WORKERS)
    if base_score is None:
        base_score = r["score"]
    nodes = sum(x["num_pred_nodes"] for x in r["per_sample"])
    rows.append((tag, r, nodes))
    print(f"{tag:12s} {r['score']:9.5f} {r['score']-base_score:+9.5f} "
          f"{r['adj_edge_jaccard']:9.5f} {r['division_jaccard']:7.4f} {nodes:9,}  "
          f"({r['division_tp']}/{r['division_fp']}/{r['division_fn']})  [{time.time()-t0:.0f}s]",
          flush=True)

best = max(rows, key=lambda x: x[1]["score"])
print(f"\nbest upstream config: {best[0]}  ({best[1]['score']:.5f}, "
      f"{best[1]['score']-base_score:+.5f} vs base)")
