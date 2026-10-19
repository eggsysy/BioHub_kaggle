"""Do the upstream winners stack with the tuned division config?

bidir0.30 and det0.980 were each measured under the SHIPPED post-processing.
The division config also targets divisions, so its gain may overlap with
bidir0.30's. Post-processing is offline, so this can be checked on each
upstream cache without another GPU run.
"""
import sys, json
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

SWEEP = Path("cache/sweep_cache")
DIV = json.loads(Path("harness/best_division.json").read_text())
stems = json.loads((SWEEP / "manifest.json").read_text())["stems"]

print(f"{len(stems)} samples; 'shipped' = 0.935 post-proc, 'division' = tuned\n")
print(f"{'upstream':12s} {'postproc':10s} {'score':>9s} {'vs base/shipped':>16s} "
      f"{'adj_edge':>9s} {'div_J':>7s}  div(tp/fp/fn)")

ref = None
for tag in ["base", "bidir0.30", "det0.980"]:
    d = SWEEP / tag
    for label, post in (("shipped", {}), ("division", DIV)):
        r = replay.score_stems_parallel(stems, post, d, workers=5)
        if ref is None:
            ref = r["score"]
        print(f"{tag:12s} {label:10s} {r['score']:9.5f} {r['score']-ref:+16.5f} "
              f"{r['adj_edge_jaccard']:9.5f} {r['division_jaccard']:7.4f}  "
              f"({r['division_tp']}/{r['division_fp']}/{r['division_fn']})", flush=True)
