# ============================================================
# CELL 9 (NEW) -- VALIDATOR: scoring against the official metric
# (https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
# Division-matching reflects the organizers' post-exploit patch (single
# connected component + both-daughter-lineage coverage + anchor + fork).
# ============================================================

from scipy.optimize import linear_sum_assignment


def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        voxel_scale = np.array(VOXEL_SCALE_UM, dtype=float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * voxel_scale
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * voxel_scale
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        BIG = 1e6
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]
    return pred_to_gt, gt_to_pred


def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = (mt is not None and mt in gt_incoming_source) or (
            ms is not None and bool(gt_outgoing.get(ms))
        )
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return tp, fp, fn


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))


def weakly_connected_components(node_ids, edges):
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t in edges:
        if s in parent and t in parent:
            union(s, t)
    return {n: find(n) for n in node_ids}


def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s

    pred_out: dict[int, set[int]] = {}
    for s, t in pred_edges:
        pred_out.setdefault(s, set()).add(t)

    pred_node_ids = list(pred_nodes.keys())
    pred_edge_list = list(pred_edges)
    components = weakly_connected_components(pred_node_ids, pred_edge_list)
    fork_components = {
        components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components
    }
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    def lineage_descendants(root_child: int) -> set[int]:
        seen = {root_child}
        stack = [root_child]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = 0
    fn = 0
    tp_gt_sources: set[int] = set()

    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])
        if len(children) < 2:
            continue
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]

        lineage_hit_components: list[set[int]] = []
        ok = True
        for child in children[:2]:
            lineage = lineage_descendants(child)
            hit_comp_ids = {
                components[p_id]
                for gt_id in lineage
                if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components
            }
            if not hit_comp_ids:
                ok = False
                break
            lineage_hit_components.append(hit_comp_ids)

        if not ok or not anchor_pred_nodes:
            fn += 1
            continue

        anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
        if not anchor_comp_ids:
            fn += 1
            continue

        found = any(
            comp_id in lineage_hit_components[0]
            and comp_id in lineage_hit_components[1]
            and comp_id in fork_components
            for comp_id in anchor_comp_ids
        )
        if found:
            tp += 1
            tp_gt_sources.add(gsrc)
        else:
            fn += 1

    fp = 0
    for n, outs in pred_out.items():
        if len(outs) < 2:
            continue
        g = pred_to_gt.get(n)
        if g is None or g not in gt_out or g in tp_gt_sources:
            continue
        fp += 1

    return tp, fp, fn


def decompose_errors(pred_nodes, gt_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    """Splits error mass into detection vs. fragmentation vs. wrong-association,
    using the exact same pred_to_gt/gt_to_pred matching compute_edge_confusion
    uses. Division errors are already isolated by compute_division_confusion;
    this covers everything else -- the diagnostic breakdown for deciding
    whether further gains are in detection, linking, or fragmentation."""
    gt_edge_set = set(gt_edges)
    pred_edge_set = set(pred_edges)
    gt_outgoing: dict[int, set[int]] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)

    missed_gt_nodes = sum(1 for g in gt_nodes if g not in gt_to_pred)
    spurious_pred_nodes = sum(1 for p in pred_nodes if p not in pred_to_gt)

    recovered = fragmented = lost_to_detection = 0
    for gs, gtid in gt_edge_set:
        ps, pt = gt_to_pred.get(gs), gt_to_pred.get(gtid)
        if ps is None or pt is None:
            lost_to_detection += 1
        elif (ps, pt) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1

    wrong_association = 0
    for ps, pt in pred_edge_set:
        ms, mt = pred_to_gt.get(ps), pred_to_gt.get(pt)
        if ms is not None and mt is not None and mt not in gt_outgoing.get(ms, ()):
            wrong_association += 1

    return {
        "missed_gt_nodes": missed_gt_nodes,
        "spurious_pred_nodes": spurious_pred_nodes,
        "edges_recovered": recovered,
        "edges_fragmented": fragmented,
        "edges_lost_to_detection": lost_to_detection,
        "wrong_association_edges": wrong_association,
    }


def _find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def read_estimated_true_node_count(geff_path: Path):
    for candidate in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        found = _find_key_recursive(payload, "estimated_number_of_nodes")
        if found is not None:
            try:
                return float(found)
            except (TypeError, ValueError):
                continue
    return None


def graph_to_plain(graph):
    nodes: dict[int, tuple] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = (int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"]))
    edges: list[tuple[int, int]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edges.append((int(row["source_id"]), int(row["target_id"])))
    return nodes, edges


def nodes_by_id_to_plain(nodes_by_id):
    return {nid: (int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"])) for nid, n in nodes_by_id.items()}


def score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true):
    p2g, g2p = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    tp, fp, fn = compute_edge_confusion(pred_edges_plain, gt_edges_plain, p2g, g2p)
    jac = edge_jaccard(tp, fp, fn)
    t_pred = len(pred_nodes_plain)
    adj = adjusted_jaccard(jac, t_pred, t_true, a=VALIDATOR_NODE_COUNT_PENALTY_A)
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, p2g, g2p
    )
    errors = decompose_errors(pred_nodes_plain, gt_nodes_plain, pred_edges_plain, gt_edges_plain, p2g, g2p)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    row = {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn, "edge_jaccard": jac,
        "t_pred": t_pred, "t_true": t_true, "adjusted_edge_jaccard": adj,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn, "div_jaccard": div_jac,
        "weight": tp + fp + fn,
    }
    row.update(errors)
    return row


def aggregate_official(sample_rows):
    total_w = sum(r["weight"] for r in sample_rows) or 1
    weighted_adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in sample_rows) / total_w
    div_tp = sum(r["div_tp"] for r in sample_rows)
    div_fp = sum(r["div_fp"] for r in sample_rows)
    div_fn = sum(r["div_fn"] for r in sample_rows)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    return {
        "adjusted_edge_jaccard": weighted_adj,
        "division_jaccard": div_jac,
        "proxy_score": weighted_adj + VALIDATOR_DIVISION_WEIGHT * div_jac,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn,
        "missed_gt_nodes": sum(r["missed_gt_nodes"] for r in sample_rows),
        "spurious_pred_nodes": sum(r["spurious_pred_nodes"] for r in sample_rows),
        "edges_recovered": sum(r["edges_recovered"] for r in sample_rows),
        "edges_fragmented": sum(r["edges_fragmented"] for r in sample_rows),
        "edges_lost_to_detection": sum(r["edges_lost_to_detection"] for r in sample_rows),
        "wrong_association_edges": sum(r["wrong_association_edges"] for r in sample_rows),
    }


validator_sample_rows: list[dict[str, object]] = []
validator_summary_rows: list[dict[str, object]] = []

if VALIDATOR_ENABLE and val_stems:
    val_pred_paths = {
        stem: found
        for stem in val_stems
        if (found := next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)) is not None
    }
    missing = [s for s in val_stems if s not in val_pred_paths]
    if missing:
        print(f"VALIDATOR: no prediction .geff found for {missing} -- did Cell 8 run and succeed?")

    rows_this_config = []
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        pred_path = val_pred_paths.get(stem)
        if not gt_path.exists() or pred_path is None:
            print(f"VALIDATOR: skipping {stem} (missing GT or prediction .geff)")
            continue
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)

        pred_graph = graph_from_geff(pred_path)
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        # REVIEW FIX: DeepCenter's gap/division veto path reads raw zarr
        # volume data via read_test_frame(dataset, t, ...), which resolves
        # TEST_DIR as a bare global at call time -- correct for the real
        # test-set run, but held-out validator samples live in TRAIN_DIR.
        # Redirect the global for exactly the duration of this call and
        # restore it unconditionally, even if filter_output_graph raises.
        _real_test_dir = TEST_DIR
        globals()["TEST_DIR"] = TRAIN_DIR
        try:
            processed_nodes, processed_edges, _stage_stats = filter_output_graph(
                raw_nodes_by_id, raw_edges, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
        finally:
            globals()["TEST_DIR"] = _real_test_dir
        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
        pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]

        row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
        row["stem"] = stem
        row["t_true_source"] = "estimated_number_of_nodes" if t_true is not None else "MISSING"
        rows_this_config.append(row)
        validator_sample_rows.append(row)

    if rows_this_config:
        summary = aggregate_official(rows_this_config)
        summary["n_samples"] = len(rows_this_config)
        validator_summary_rows.append(summary)

    print()
    print("=" * 78)
    print("LOCAL VALIDATOR -- proxy score using the official metric formula")
    print("(https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)")
    print("=" * 78)
    for row in validator_sample_rows:
        print(f"  {row['stem']:<28} edge_jaccard={row['edge_jaccard']:.4f} "
              f"adj_edge_jaccard={row['adjusted_edge_jaccard']:.4f} T_pred={row['t_pred']} "
              f"T_true={row['t_true']} div(tp/fp/fn)=({row['div_tp']}/{row['div_fp']}/{row['div_fn']})")
        print(f"      errors: missed_gt_nodes={row['missed_gt_nodes']} "
              f"spurious_pred_nodes={row['spurious_pred_nodes']} "
              f"recovered={row['edges_recovered']} fragmented={row['edges_fragmented']} "
              f"lost_to_detection={row['edges_lost_to_detection']} "
              f"wrong_association={row['wrong_association_edges']}")
    print()
    for summary in validator_summary_rows:
        print(f"n={summary['n_samples']}  adjusted_edge_jaccard={summary['adjusted_edge_jaccard']:.4f}  "
              f"division_jaccard={summary['division_jaccard']:.4f}  "
              f"PROXY_SCORE={summary['proxy_score']:.4f}")
        print(f"  totals -- missed_gt_nodes={summary['missed_gt_nodes']} "
              f"spurious_pred_nodes={summary['spurious_pred_nodes']} "
              f"recovered={summary['edges_recovered']} fragmented={summary['edges_fragmented']} "
              f"lost_to_detection={summary['edges_lost_to_detection']} "
              f"wrong_association={summary['wrong_association_edges']}")

    if validator_sample_rows:
        with VALIDATOR_STATS_PATH.open("w", newline="") as f:
            fieldnames = sorted({k for row in validator_sample_rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in validator_sample_rows:
                writer.writerow(row)
        print(f"\nPer-sample validator rows written to {VALIDATOR_STATS_PATH}")
else:
    print("VALIDATOR: disabled or no held-out samples available -- skipping scoring.")
