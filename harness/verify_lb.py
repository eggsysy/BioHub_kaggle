"""Does the 50-sample cache reproduce the leaderboard's ranking?

We have three configs with known hidden-test scores:

    baseline        0.935
    widened gates   0.912   (-0.023)
    safe-div off    0.908   (-0.027)

If the cache ranks them the same way and with roughly proportional gaps, it
can be trusted to drive a search. If it does not, no amount of Optuna on it
is worth running. This is the check the 4-sample mini-cache failed.
"""
import sys, time
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

CACHE = Path("cache/pred_cache")
stems = replay.discover_stems(CACHE)

CONFIGS = {
    "baseline (LB 0.935)": {},
    "widened gates (LB 0.912)": {
        "BIOHUB_SAFE_DIV_MAX_UM": "11.8", "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "15.3",
        "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "0",
        "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.015", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.006",
    },
    "safe-div off (LB 0.908)": {"BIOHUB_OUTPUT_SAFE_DIVISIONS": "0"},
}
LB = {"baseline (LB 0.935)": 0.935, "widened gates (LB 0.912)": 0.912,
      "safe-div off (LB 0.908)": 0.908}

base_local = None
print(f"{'config':28s} {'cache':>9s} {'d_cache':>9s} {'d_LB':>8s} {'div_J':>7s}  div(tp/fp/fn)")
for name, ov in CONFIGS.items():
    t = time.time()
    r = replay.score_stems_parallel(stems, ov, CACHE, workers=10)
    if base_local is None:
        base_local = r["score"]
    d_cache = r["score"] - base_local
    d_lb = LB[name] - 0.935
    print(f"{name:28s} {r['score']:9.5f} {d_cache:+9.5f} {d_lb:+8.3f} "
          f"{r['division_jaccard']:7.4f}  ({r['division_tp']}/{r['division_fp']}/{r['division_fn']})"
          f"   [{time.time()-t:.0f}s]", flush=True)
