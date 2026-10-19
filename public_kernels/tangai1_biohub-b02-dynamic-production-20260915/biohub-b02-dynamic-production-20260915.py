#!/usr/bin/env python3
"""Hidden-test-safe B02 production submission entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


COMPETITION = "biohub-cell-tracking-during-development"
RUNTIME_MARKER = "B02_RUNTIME_MANIFEST.json"


def _find_runtime() -> Path:
    configured = os.environ.get("BIOHUB_B02_RUNTIME_ROOT")
    if configured:
        root = Path(configured)
        if (root / RUNTIME_MARKER).is_file():
            return root
        raise FileNotFoundError(f"invalid BIOHUB_B02_RUNTIME_ROOT: {root}")
    roots = [Path("/kaggle/input"), Path("/tmp")]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        matches.extend(path.parent for path in root.rglob(RUNTIME_MARKER))
    if not matches:
        raise FileNotFoundError(f"{RUNTIME_MARKER} not found")
    return sorted(set(matches), key=lambda path: str(path))[0]


def _find_competition_root() -> Path:
    configured = os.environ.get("BIOHUB_B02_COMP_ROOT")
    if configured:
        root = Path(configured)
        if (root / "test").is_dir():
            return root
        raise FileNotFoundError(f"invalid BIOHUB_B02_COMP_ROOT: {root}")
    candidates = [
        Path(f"/kaggle/input/competitions/{COMPETITION}"),
        Path(f"/kaggle/input/{COMPETITION}"),
    ]
    for root in candidates:
        if (root / "test").is_dir():
            return root
    raise FileNotFoundError("competition test directory is not mounted")


def _ensure_offline_packages() -> None:
    """Use only mounted wheels if the pinned Kaggle image lacks data packages."""
    try:
        import zarr  # noqa: F401
        import tracksdata  # noqa: F401
        return
    except ImportError:
        pass
    wheel_dirs = []
    for root in Path("/kaggle/input").glob("**/wheels"):
        if root.is_dir():
            wheel_dirs.append(root)
    if not wheel_dirs:
        raise RuntimeError("zarr/tracksdata unavailable and no offline wheel directory mounted")
    packages = ["zarr", "tracksdata", "geff", "geff-spec", "numcodecs", "blosc2"]
    for wheel_dir in sorted(set(wheel_dirs), key=lambda path: str(path)):
        command = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheel_dir)] + packages
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return
    raise RuntimeError("failed to install required packages from mounted offline wheels")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("biohub_b02_dynamic_production", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_submission(path: Path, test_dir: Path, report_path: Path) -> dict[str, object]:
    import csv
    import zarr

    expected_header = [
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id",
    ]
    expected_datasets = sorted(item.stem for item in test_dir.glob("*.zarr"))
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise RuntimeError({"header": reader.fieldnames, "expected": expected_header})
        rows = list(reader)
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("submission id column is not consecutive")
    actual_datasets = sorted({row["dataset"] for row in rows})
    if actual_datasets != expected_datasets:
        raise RuntimeError({"dataset_coverage": actual_datasets, "expected": expected_datasets})

    per_dataset = {}
    for dataset in expected_datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        nodes = [row for row in group if row["row_type"] == "node"]
        edges = [row for row in group if row["row_type"] == "edge"]
        if len(nodes) == 0:
            raise RuntimeError(f"{dataset}: no nodes")
        node_ids = [int(row["node_id"]) for row in nodes]
        if sorted(node_ids) != list(range(1, len(nodes) + 1)):
            raise RuntimeError(f"{dataset}: node IDs are not exactly one-based")
        node_t = {int(row["node_id"]): int(row["t"]) for row in nodes}
        meta = json.loads((test_dir / f"{dataset}.zarr" / "0" / "zarr.json").read_text())
        shape = tuple(int(value) for value in meta["shape"])
        indegree = {}
        outdegree = {}
        edge_keys = set()
        for row in nodes:
            if any(int(row[key]) < 0 for key in ("t", "z", "y", "x")):
                raise RuntimeError(f"{dataset}: negative node coordinate")
            if not (0 <= int(row["t"]) < shape[0] and 0 <= int(row["z"]) < shape[1]
                    and 0 <= int(row["y"]) < shape[2] and 0 <= int(row["x"]) < shape[3]):
                raise RuntimeError(f"{dataset}: node coordinate out of bounds")
        for row in edges:
            source = int(row["source_id"])
            target = int(row["target_id"])
            if source not in node_t or target not in node_t:
                raise RuntimeError(f"{dataset}: edge references missing node")
            if (source, target) in edge_keys:
                raise RuntimeError(f"{dataset}: duplicate edge")
            edge_keys.add((source, target))
            if node_t[target] != node_t[source] + 1:
                raise RuntimeError(f"{dataset}: non-adjacent edge")
            indegree[target] = indegree.get(target, 0) + 1
            outdegree[source] = outdegree.get(source, 0) + 1
        if max(indegree.values(), default=0) > 1 or max(outdegree.values(), default=0) > 2:
            raise RuntimeError(f"{dataset}: invalid lineage degree")
        per_dataset[dataset] = {
            "nodes": len(nodes),
            "edges": len(edges),
            "node_id_min": min(node_ids),
            "node_id_max": max(node_ids),
            "shape": shape,
            "max_indegree": max(indegree.values(), default=0),
            "max_outdegree": max(outdegree.values(), default=0),
        }
    report = {
        "status": "PASS",
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "datasets": per_dataset,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("B02_DYNAMIC_AUDIT:", json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> int:
    _ensure_offline_packages()
    runtime = _find_runtime()
    competition = _find_competition_root()
    test_dir = competition / "test"
    output_dir = Path(os.environ.get("BIOHUB_B02_OUTPUT_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)
    production = _load_module(runtime / "experiments/20260914_v6_b02_biohub_native_ssl/production_infer.py")
    work_root = Path(os.environ.get("BIOHUB_B02_WORK_ROOT", "/tmp/biohub_b02_dynamic_root"))
    production.ROOT = work_root
    production.EXP = work_root / "experiments/20260914_v6_b02_biohub_native_ssl"
    production.TEST_DIR = test_dir
    production.TMP_DIR = Path(os.environ.get("BIOHUB_B02_TMP_DIR", "/tmp/biohub_v6_b02_dynamic"))
    production.REPORT_PATH = output_dir / "b02_production_inference.json"
    production.CANDIDATE_PATH = output_dir / "submission.csv"
    production.B01_PATH = runtime / "experiments/20260913_v6_b01_long_window_temporal/run.py"
    production.C46_PATH = runtime / "experiments/20260903_c46_production/run.py"
    production.PRIMARY = runtime / "weights/v6_b02_ssl/b02_primary_production_all199.pth"
    production.SECONDARY = runtime / "weights/v6_b02_ssl/b02_secondary_production_all199.pth"
    if not production.PRIMARY.is_file() or not production.SECONDARY.is_file():
        raise FileNotFoundError("B02 production weights are missing from runtime dataset")
    print(json.dumps({
        "stage": "B02_DYNAMIC_START",
        "runtime": str(runtime),
        "test_dir": str(test_dir),
        "datasets": sorted(path.stem for path in test_dir.glob("*.zarr")),
        "primary_sha256": _sha256(production.PRIMARY),
        "secondary_sha256": _sha256(production.SECONDARY),
    }, sort_keys=True), flush=True)
    result = production.main()
    _audit_submission(production.CANDIDATE_PATH, test_dir, output_dir / "b02_dynamic_audit.json")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
