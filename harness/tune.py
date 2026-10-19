"""Optuna search over the post-processing knobs, scored by the official metric.

Runs entirely on cached prediction graphs, so a trial costs seconds rather
than a Kaggle GPU notebook. Search spaces are grouped into stages because the
knobs are not independent: division geometry, track filtering and gap closing
each have their own interacting cluster, and tuning one cluster at a time on a
frozen remainder is both faster and far less prone to overfitting 50 samples.

Usage:
    python harness/tune.py --stage division --trials 300
    python harness/tune.py --stage all --trials 800 --out best_division.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import optuna  # noqa: E402
import replay  # noqa: E402


def _space_division(trial) -> dict[str, str]:
    """Division geometry + the DeepCenter veto that gates it.

    This is the cluster with the most headroom. Measured against all 151
    annotated divisions (see gt_divisions.py), the shipped gates admit only
    23.8% of real divisions, which caps division recall before any model
    decision is made. Bounds here are widened to cover the measured p95
    geometry: parent<=11.8um, sister<=15.3um, divergence effectively off.
    """
    cfg = {
        "BIOHUB_SAFE_DIV_MAX_UM": trial.suggest_float("safe_div_max_um", 4.0, 16.0),
        "BIOHUB_SAFE_DIV_SISTER_MAX_UM": trial.suggest_float("sister_max_um", 6.0, 22.0),
        "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": trial.suggest_float("existing_child_um", 4.0, 18.0),
        "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": trial.suggest_float("frame_frac_cap", 0.001, 0.05, log=True),
        "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": trial.suggest_float("global_frac_cap", 0.001, 0.03, log=True),
        "BIOHUB_SAFE_DIV_DIVERGE_UM": trial.suggest_float("diverge_um", -2.0, 5.0),
        "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": str(int(trial.suggest_categorical("require_divergence", [0, 1]))),
        "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": str(int(trial.suggest_categorical("require_mutual_nn", [0, 1]))),
        "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": str(int(trial.suggest_categorical("dc_safe_div_veto", [0, 1]))),
        "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": trial.suggest_float("dc_safe_div_thr", 0.0, 0.6),
        "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": str(int(trial.suggest_categorical("keep_div_components", [0, 1]))),
    }
    return {k: str(v) for k, v in cfg.items()}


def _space_tracks(trial) -> dict[str, str]:
    """Track-length filtering and the node-count adjustment it drives.

    Dropping short components lowers T_pred; because unmatched predicted nodes
    are not false positives, that can be a net gain through the node-count
    term even when it costs a little edge recall.
    """
    cfg = {
        "BIOHUB_OUTPUT_MIN_TRACK_LEN": trial.suggest_int("min_track_len", 2, 14),
        "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": str(int(trial.suggest_categorical("filter_short", [0, 1]))),
        "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": str(int(trial.suggest_categorical("rescue", [0, 1]))),
        "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": trial.suggest_int("rescue_min_len", 2, 8),
        "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": trial.suggest_float("rescue_min_prob", 0.5, 0.98),
        "BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC": trial.suggest_float("rescue_max_frac", 0.002, 0.06, log=True),
        "BIOHUB_OUTPUT_PRUNE_ISOLATED": str(int(trial.suggest_categorical("prune_isolated", [0, 1]))),
    }
    return {k: str(v) for k, v in cfg.items()}


def _space_gaps(trial) -> dict[str, str]:
    """Gap closing and relinking. Approximate offline (synthetic midpoints use
    a nearest-neighbour DeepCenter fallback) -- confirm winners on Kaggle."""
    cfg = {
        "BIOHUB_GAP_CLOSE_MAX_GAP": trial.suggest_int("gap_max", 1, 3),
        "BIOHUB_GAP_CLOSE_UM": trial.suggest_float("gap_um", 3.5, 9.0),
        "BIOHUB_GAP_CLOSE_REUSE_UM": trial.suggest_float("gap_reuse_um", 1.5, 5.5),
        "BIOHUB_GAP2_MAX_TOTAL_UM": trial.suggest_float("gap2_total_um", 6.0, 15.0),
        "BIOHUB_GAP2_MAX_STEP_UM": trial.suggest_float("gap2_step_um", 2.5, 7.0),
        "BIOHUB_GAP2_MAX_LINKS_FRAC": trial.suggest_float("gap2_links_frac", 0.001, 0.02, log=True),
        "BIOHUB_MOTION_RELINK_TIGHT_UM": trial.suggest_float("relink_tight_um", 3.5, 9.0),
        "BIOHUB_MOTION_RELINK_RELAXED_UM": trial.suggest_float("relink_relaxed_um", 6.0, 14.0),
        "BIOHUB_MOTION_RELINK_LEARNED_BONUS": trial.suggest_float("relink_bonus", 0.0, 2.0),
        "BIOHUB_OUTPUT_EDGE_MAX_UM": trial.suggest_float("edge_max_um", 8.0, 20.0),
    }
    return {k: str(v) for k, v in cfg.items()}


class _ProbeTrial:
    """Minimal stand-in that lets a search space be evaluated for its key set."""

    def suggest_float(self, name, low, high, **kw):
        return low

    def suggest_int(self, name, low, high, **kw):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


SPACES = {
    "division": _space_division,
    "tracks": _space_tracks,
    "gaps": _space_gaps,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*SPACES, "all"], default="division")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--cache", type=Path, default=replay.CACHE_DIR)
    ap.add_argument("--frozen", type=Path, default=None,
                    help="JSON of overrides to hold fixed (e.g. a previous stage's winner)")
    ap.add_argument("--out", type=Path, default=HARNESS / "best_config.json")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N samples")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    stems = replay.discover_stems(args.cache)
    if args.limit:
        stems = stems[: args.limit]
    print(f"searching over {len(stems)} cached samples ({args.workers} workers)")

    # Fail fast on knobs that cannot move the offline score.
    for name, space in SPACES.items():
        try:
            replay.assert_effective(space(_ProbeTrial()))
        except ValueError as exc:
            raise SystemExit(f"search space '{name}' is broken: {exc}")
    print("search spaces verified: every knob reaches the post-processing stage")

    frozen = json.loads(args.frozen.read_text()) if args.frozen else {}
    if frozen:
        print(f"holding {len(frozen)} knobs fixed from {args.frozen}")

    base = replay.score_stems_parallel(stems, frozen, args.cache, args.workers)
    print(f"\nBASELINE  score={base['score']:.5f}  "
          f"adj_edge={base['adj_edge_jaccard']:.5f}  div_J={base['division_jaccard']:.4f} "
          f"(tp/fp/fn = {base['division_tp']}/{base['division_fp']}/{base['division_fn']})")
    print(f"DeepCenter cache: {base['dc_counters']}\n")

    spaces = list(SPACES.values()) if args.stage == "all" else [SPACES[args.stage]]

    def objective(trial):
        overrides = dict(frozen)
        for space in spaces:
            overrides.update(space(trial))
        try:
            return replay.score_stems_parallel(
                overrides=overrides, stems=stems,
                cache_dir=args.cache, workers=args.workers,
            )["score"]
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({}, skip_if_exists=True)  # evaluate the shipped config first
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBEST  score={study.best_value:.5f}   (baseline {base['score']:.5f}, "
          f"delta {study.best_value - base['score']:+.5f})")
    for k, v in study.best_params.items():
        print(f"   {k} = {v}")

    best_overrides = dict(frozen)
    trial = study.best_trial
    for space in spaces:
        best_overrides.update(space(optuna.trial.FixedTrial(trial.params)))
    args.out.write_text(json.dumps(best_overrides, indent=2))
    study.trials_dataframe().to_csv(args.out.with_suffix(".trials.csv"), index=False)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.trials.csv')}")


if __name__ == "__main__":
    main()
