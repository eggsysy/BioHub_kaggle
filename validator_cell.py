# ============================================================
# CELL 8 (NEW) -- VALIDATOR: held-out sample selection + prediction run
# ============================================================
# Ported from the team's own local-validator harness onto this notebook's
# own path variables and its own (genuinely parallel, Popen-based) shard
# runner -- nothing in Cells 1-7 is touched, so this notebook still
# reproduces 0.915 if VALIDATOR_ENABLE is turned off.

TRAIN_DIR = COMP_DIR / "train"

VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1") != "0"
VALIDATOR_N_PER_TYPE = int(os.environ.get("BIOHUB_VALIDATOR_N_PER_TYPE", "2"))
VALIDATOR_MATCH_RADIUS_UM = float(os.environ.get("BIOHUB_VALIDATOR_MATCH_RADIUS_UM", "7.0"))
VALIDATOR_NODE_COUNT_PENALTY_A = float(os.environ.get("BIOHUB_VALIDATOR_NODE_COUNT_PENALTY_A", "0.1"))
VALIDATOR_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_VALIDATOR_DIVISION_WEIGHT", "0.1"))
VALIDATOR_STATS_PATH = WORKING_DIR / "validator_results.csv"

val_stems: list[str] = []
if VALIDATOR_ENABLE and TRAIN_DIR.exists():
    train_stems_all = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
    test_stem_set = set(test_stems)  # from Cell 6 -- guards against train/test leakage
    overlap = [s for s in train_stems_all if s in test_stem_set]
    if overlap:
        print(f"VALIDATOR: excluding {len(overlap)} TRAIN stem(s) that also appear in TEST_DIR: {overlap}")
    candidates = [s for s in train_stems_all if s not in test_stem_set]

    # REVIEW: division-aware sample selection. GT divisions are annotated in
    # only ~44% of videos, so plain alphabetical selection risks picking a
    # validator set that structurally can't show division_jaccard movement.
    # This checks each candidate's own GT graph for a division (out-degree>=2)
    # -- a computed, per-video property, not a hardcoded list of stem names --
    # so it generalizes to any future TRAIN_DIR contents rather than
    # overfitting to today's specific videos.
    def _stem_has_gt_division(stem: str) -> bool:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        try:
            graph = graph_from_geff(gt_path)
        except Exception:
            return False
        out_degree: dict[int, int] = {}
        for row in graph.edge_attrs().iter_rows(named=True):
            s = int(row["source_id"])
            out_degree[s] = out_degree.get(s, 0) + 1
        return any(d >= 2 for d in out_degree.values())

    by_prefix: dict[str, list[str]] = {}
    for s in candidates:
        by_prefix.setdefault(s.split("_")[0], []).append(s)

    division_flags: dict[str, bool] = {}
    for prefix, stems in by_prefix.items():
        for s in stems:
            division_flags[s] = _stem_has_gt_division(s)

    for prefix, stems in sorted(by_prefix.items()):
        ranked = sorted(stems, key=lambda s: (not division_flags[s], s))
        val_stems.extend(ranked[:VALIDATOR_N_PER_TYPE])
    n_division_selected = sum(1 for s in val_stems if division_flags[s])
    print(f"VALIDATOR: selected {len(val_stems)} held-out TRAIN samples "
          f"({VALIDATOR_N_PER_TYPE} per embryo-type prefix, {len(by_prefix)} prefixes found, "
          f"{n_division_selected} contain a GT division)")
    print(val_stems)
elif VALIDATOR_ENABLE:
    print(f"VALIDATOR: TRAIN_DIR not found at {TRAIN_DIR} -- skipping.")
else:
    print("VALIDATOR: disabled (BIOHUB_VALIDATOR_ENABLE=0).")


def _merge_validator_shards(worker_count: int, stems: list[str], method_prefix: str) -> Path:
    """Same logic as _merge_prediction_shards (Cell 6), parameterized for an
    arbitrary stem list and method prefix instead of the global test_stems/
    METHOD -- that function is hardcoded to the real test run and isn't
    safe to call directly for a different sample set."""
    import shutil as _shutil
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(stems)

    for shard_index in range(worker_count):
        shard_method = f"{method_prefix}_gpu{shard_index}"
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(stems[shard_index::worker_count])
        found = {p.stem for p in sorted(shard_dir.glob("*.geff"))}
        if found != expected:
            raise RuntimeError(
                f"VALIDATOR shard {shard_index} output mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        overlap_ds = seen & found
        if overlap_ds:
            raise RuntimeError(f"VALIDATOR: duplicate datasets across shards: {sorted(overlap_ds)}")
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(
            f"VALIDATOR: merged shards do not cover the held-out set: "
            f"missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}"
        )

    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f"VALIDATOR: shards used inconsistent prediction roots: {username_roots}")

    final_root = next(iter(username_roots)) / method_prefix
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_val_staging"
    if staging_dir.exists():
        _shutil.rmtree(staging_dir) if staging_dir.is_dir() else staging_dir.unlink()
    staging_dir.mkdir(parents=True, exist_ok=False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob("*.geff")):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f"VALIDATOR: refusing to overwrite duplicate output: {destination}")
            _shutil.move(str(source), str(destination))

    merged = {p.stem for p in staging_dir.glob("*.geff")}
    if merged != expected_all:
        raise RuntimeError(
            f"VALIDATOR: staged directory failed verification: "
            f"missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}"
        )

    if final_dir.exists():
        _shutil.rmtree(final_dir) if final_dir.is_dir() else final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    print(f"VALIDATOR: merged {len(merged)} prediction graphs into {final_dir}")
    return final_dir


predict_val_seconds = None
if VALIDATOR_ENABLE and val_stems:
    val_splits_path = REPO_DIR / "kaggle_val_splits.json"
    val_splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": val_stems}], indent=2))
    val_method_prefix = f"{METHOD}_val"

    predict_val_cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", str(TRAIN_DIR),
        "--splits", str(val_splits_path.name),
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        predict_val_cmd.append("--use-ilp")

    _val_start = time.time()
    val_worker_count = min(2, _torch.cuda.device_count(), len(val_stems))
    if val_worker_count >= 2:
        cuda_tokens = _visible_cuda_tokens(val_worker_count)
        val_processes: dict[int, subprocess.Popen] = {}
        val_commands: dict[int, list[str]] = {}
        print(f"VALIDATOR: launching {val_worker_count} shards on CUDA devices {cuda_tokens}")
        for shard_index in range(val_worker_count):
            shard_cmd = [*predict_val_cmd, "--method", f"{val_method_prefix}_gpu{shard_index}",
                         "--slice", f"{shard_index}::{val_worker_count}"]
            shard_env = {**os.environ, "PYTHONPATH": "src"}
            shard_env["CUDA_VISIBLE_DEVICES"] = cuda_tokens[shard_index]
            val_commands[shard_index] = shard_cmd
            val_processes[shard_index] = subprocess.Popen(shard_cmd, cwd=REPO_DIR, env=shard_env)
        _wait_for_prediction_shards(val_processes, val_commands)
        _merge_validator_shards(val_worker_count, val_stems, val_method_prefix)
    else:
        print("VALIDATOR: using single-process prediction (fewer than 2 GPUs or samples).")
        subprocess.run([*predict_val_cmd, "--method", val_method_prefix],
                        cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
    predict_val_seconds = time.time() - _val_start
    print(f"VALIDATOR: prediction completed in {predict_val_seconds / 60:.2f} minutes")
