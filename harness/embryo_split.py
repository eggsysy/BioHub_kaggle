"""Does the tuned division config generalise across embryos?

Train and test are embryo-disjoint, so a config that only helps one embryo is
fitting noise. 11 free parameters were tuned against 73 divisions over 50
samples; this scores the winner separately on the 25 44b6 and 25 6bba samples.
A real improvement should show on both.
"""
import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

CACHE = Path("cache/pred_cache")
stems = replay.discover_stems(CACHE)
groups = {
    "44b6 (n=%d)" % sum(1 for s in stems if s.startswith("44b6")): [s for s in stems if s.startswith("44b6")],
    "6bba (n=%d)" % sum(1 for s in stems if s.startswith("6bba")): [s for s in stems if s.startswith("6bba")],
    "all   (n=%d)" % len(stems): stems,
}

TRIAL71 = {
    "BIOHUB_SAFE_DIV_MAX_UM": "9.250193043603588",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "19.486653287190908",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "15.236789836845439",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.001706745704996418",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.0022919771899392125",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "3.686568907789547",
    "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "1",
    "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "1",
    "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.18808779448112303",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "0",
}

TRIAL14 = {
    "BIOHUB_SAFE_DIV_MAX_UM": "9.24947641491875",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "21.681973290746676",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "14.636698787624212",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0010686766991159639",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.013221089871343152",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "3.716168707984469",
    "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "1",
    "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "1",
    "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.3029993243178887",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "0",
}

print(f"{'group':14s} {'config':10s} {'score':>9s} {'delta':>9s} {'adj_edge':>9s} {'div_J':>7s}  div(tp/fp/fn)")
for gname, gstems in groups.items():
    base = replay.score_stems_parallel(gstems, {}, CACHE, workers=4)
    tuned = replay.score_stems_parallel(gstems, TRIAL14, CACHE, workers=4)
    t71 = replay.score_stems_parallel(gstems, TRIAL71, CACHE, workers=10)
    for label, r, d in (("baseline", base, 0.0),
                        ("trial14", tuned, tuned["score"] - base["score"]),
                        ("trial71", t71, t71["score"] - base["score"])):
        print(f"{gname:14s} {label:10s} {r['score']:9.5f} {d:+9.5f} {r['adj_edge_jaccard']:9.5f} "
              f"{r['division_jaccard']:7.4f}  ({r['division_tp']}/{r['division_fp']}/{r['division_fn']})",
              flush=True)
