"""
Local Evaluation System for Biohub Cell Tracking
Implements the EXACT competition metric using only scipy/numpy (no tracksdata dependency).

Usage:
  On Kaggle (in the submission notebook):
    Add this as a cell before the main pipeline to calibrate on training data.
  
  Locally:
    python local_eval.py --train-dir /path/to/train --n-movies 5
"""

import numpy as np
import os
import glob
import json
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
MAX_MATCH_DIST_UM = 7.0  # Official competition cutoff
ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1


# ════════════════════════════════════════════════════════════════════
# GEFF READER (No tracksdata dependency)
# ════════════════════════════════════════════════════════════════════

def read_geff(geff_path):
    """Read ground truth from a .geff directory (Zarr v3 based)."""
    import zarr
    g = zarr.open(geff_path, mode='r')
    
    # Read nodes
    ids = np.array(g['nodes/ids'])
    t = np.array(g['nodes/props/t/values'])
    z = np.array(g['nodes/props/z/values'])
    y = np.array(g['nodes/props/y/values'])
    x = np.array(g['nodes/props/x/values'])
    
    nodes = {}
    for i in range(len(ids)):
        nodes[int(ids[i])] = {
            't': int(t[i]),
            'z': float(z[i]),
            'y': float(y[i]),
            'x': float(x[i])
        }
    
    # Read edges
    edge_ids = np.array(g['edges/ids'])  # shape (N, 2) or similar
    
    # Try different GEFF edge formats
    edges = []
    if edge_ids.ndim == 2 and edge_ids.shape[1] == 2:
        for i in range(len(edge_ids)):
            edges.append((int(edge_ids[i, 0]), int(edge_ids[i, 1])))
    else:
        # Try source/target props
        try:
            sources = np.array(g['edges/props/source_ids/values'])
            targets = np.array(g['edges/props/target_ids/values'])
            for i in range(len(sources)):
                edges.append((int(sources[i]), int(targets[i])))
        except Exception:
            pass
    
    return nodes, edges


# ════════════════════════════════════════════════════════════════════
# EXACT COMPETITION METRIC
# ════════════════════════════════════════════════════════════════════

def match_nodes_at_time(pred_nodes_t, gt_nodes_t, max_dist_um=MAX_MATCH_DIST_UM):
    """
    Bipartite matching of predicted nodes to GT nodes at a single timepoint.
    Returns: dict mapping pred_node_id -> gt_node_id
    """
    if not pred_nodes_t or not gt_nodes_t:
        return {}
    
    pred_ids = list(pred_nodes_t.keys())
    gt_ids = list(gt_nodes_t.keys())
    
    pred_coords = np.array([[pred_nodes_t[pid]['z'], pred_nodes_t[pid]['y'], pred_nodes_t[pid]['x']] for pid in pred_ids])
    gt_coords = np.array([[gt_nodes_t[gid]['z'], gt_nodes_t[gid]['y'], gt_nodes_t[gid]['x']] for gid in gt_ids])
    
    # Convert to physical µm
    pred_um = pred_coords * VOXEL_SCALE
    gt_um = gt_coords * VOXEL_SCALE
    
    # Build cost matrix
    n_pred, n_gt = len(pred_ids), len(gt_ids)
    cost = np.full((n_pred, n_gt), 1e9)
    
    for i in range(n_pred):
        for j in range(n_gt):
            d = np.linalg.norm(pred_um[i] - gt_um[j])
            if d <= max_dist_um:
                cost[i, j] = d
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matching = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_dist_um:
            matching[pred_ids[r]] = gt_ids[c]
    
    return matching


def evaluate_predictions(pred_nodes, pred_edges, gt_nodes, gt_edges, T_true_estimate=None):
    """
    Evaluate predictions against ground truth using the EXACT competition metric.
    
    pred_nodes: list of (nid, t, z, y, x)
    pred_edges: list of (source_id, target_id)
    gt_nodes: dict {nid: {t, z, y, x}} from read_geff
    gt_edges: list of (source_id, target_id) from read_geff
    T_true_estimate: coarse estimate of total true nodes (if None, uses 2x GT count)
    
    Returns: dict with edge_tp, edge_fp, edge_fn, edge_jaccard, adj_edge_jaccard, etc.
    """
    # Build pred node dict
    pred_node_dict = {}
    for nid, t, z, y, x in pred_nodes:
        pred_node_dict[nid] = {'t': int(t), 'z': float(z), 'y': float(y), 'x': float(x)}
    
    # Group nodes by timepoint
    pred_by_t = defaultdict(dict)
    gt_by_t = defaultdict(dict)
    
    for nid, data in pred_node_dict.items():
        pred_by_t[data['t']][nid] = data
    for nid, data in gt_nodes.items():
        gt_by_t[data['t']][nid] = data
    
    # Match nodes at each timepoint
    all_times = sorted(set(list(pred_by_t.keys()) + list(gt_by_t.keys())))
    pred_to_gt = {}  # pred_nid -> gt_nid
    gt_to_pred = {}  # gt_nid -> pred_nid
    
    for t in all_times:
        matching = match_nodes_at_time(pred_by_t.get(t, {}), gt_by_t.get(t, {}))
        for pid, gid in matching.items():
            pred_to_gt[pid] = gid
            gt_to_pred[gid] = pid
    
    # Build GT edge set
    gt_edge_set = set()
    gt_source_map = defaultdict(set)  # gt_source -> set of gt_targets
    gt_target_map = defaultdict(set)  # gt_target -> set of gt_sources
    
    for src, tgt in gt_edges:
        # Only count edges between consecutive timepoints
        if src in gt_nodes and tgt in gt_nodes:
            if gt_nodes[tgt]['t'] - gt_nodes[src]['t'] == 1:
                gt_edge_set.add((src, tgt))
                gt_source_map[src].add(tgt)
                gt_target_map[tgt].add(src)
    
    # Count Edge TP / FP / FN
    edge_tp = 0
    edge_fp = 0
    matched_gt_edges = set()
    
    for pred_src, pred_tgt in pred_edges:
        # Only evaluate edges between consecutive timepoints
        if pred_src in pred_node_dict and pred_tgt in pred_node_dict:
            if pred_node_dict[pred_tgt]['t'] - pred_node_dict[pred_src]['t'] != 1:
                continue
        
        gt_src = pred_to_gt.get(pred_src)
        gt_tgt = pred_to_gt.get(pred_tgt)
        
        if gt_src is not None and gt_tgt is not None and (gt_src, gt_tgt) in gt_edge_set:
            # TRUE POSITIVE
            edge_tp += 1
            matched_gt_edges.add((gt_src, gt_tgt))
        else:
            # Check if it's a FALSE POSITIVE (source or target conflict)
            is_fp = False
            
            # Source conflict: pred_src matches gt_src, but gt_src connects to different target
            if gt_src is not None and gt_src in gt_source_map:
                is_fp = True
            
            # Target conflict: pred_tgt matches gt_tgt, but gt_tgt connects from different source
            if gt_tgt is not None and gt_tgt in gt_target_map:
                is_fp = True
            
            if is_fp:
                edge_fp += 1
            # else: edge is ignored (both endpoints unmatched)
    
    edge_fn = len(gt_edge_set) - len(matched_gt_edges)
    
    # Compute Jaccard
    denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denom if denom > 0 else 0.0
    
    # Adjusted Edge Jaccard (density penalty)
    num_pred_nodes = len(pred_node_dict)
    if T_true_estimate is None:
        # Conservative estimate: 2x the GT node count (since GT is sparse)
        T_true_estimate = len(gt_nodes) * 2
    
    total_node_ratio = (num_pred_nodes - T_true_estimate) / T_true_estimate if T_true_estimate > 0 else 0
    adj_edge_jaccard = max(0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
    
    # Node recall
    node_recall = len(pred_to_gt) / len(gt_nodes) if len(gt_nodes) > 0 else 0
    
    return {
        'edge_tp': edge_tp,
        'edge_fp': edge_fp,
        'edge_fn': edge_fn,
        'edge_jaccard': edge_jaccard,
        'adj_edge_jaccard': adj_edge_jaccard,
        'node_recall': node_recall,
        'num_pred_nodes': num_pred_nodes,
        'num_gt_nodes': len(gt_nodes),
        'num_gt_edges': len(gt_edge_set),
        'total_node_ratio': total_node_ratio,
        'num_matched_nodes': len(pred_to_gt),
    }


def evaluate_on_training(train_dir, process_fn, n_movies=5):
    """
    Run the pipeline on n training movies and evaluate against GT.
    
    process_fn: function(zarr_path, ds_name) -> (nodes, edges)
    """
    zarr_dirs = sorted(glob.glob(os.path.join(train_dir, "*.zarr")))[:n_movies]
    
    totals = defaultdict(int)
    per_movie = []
    
    for zp in zarr_dirs:
        ds = os.path.basename(zp).replace(".zarr", "")
        geff_path = zp.replace(".zarr", ".geff")
        
        if not os.path.exists(geff_path):
            print(f"  [{ds}] No .geff found, skipping")
            continue
        
        print(f"\n{'='*50}")
        print(f"  Evaluating: {ds}")
        print(f"{'='*50}")
        
        # Run pipeline
        t0 = time.time()
        pred_nodes, pred_edges = process_fn(zp, ds)
        elapsed = time.time() - t0
        
        # Read GT
        gt_nodes, gt_edges = read_geff(geff_path)
        
        # Evaluate
        result = evaluate_predictions(pred_nodes, pred_edges, gt_nodes, gt_edges)
        result['time_s'] = elapsed
        result['dataset'] = ds
        per_movie.append(result)
        
        for k in ['edge_tp', 'edge_fp', 'edge_fn']:
            totals[k] += result[k]
        totals['num_pred_nodes'] += result['num_pred_nodes']
        totals['num_gt_nodes'] += result['num_gt_nodes']
        
        print(f"  Nodes: {result['num_pred_nodes']} pred, {result['num_gt_nodes']} GT, "
              f"{result['num_matched_nodes']} matched ({result['node_recall']:.1%} recall)")
        print(f"  Edges: TP={result['edge_tp']}  FP={result['edge_fp']}  FN={result['edge_fn']}")
        print(f"  Edge Jaccard: {result['edge_jaccard']:.4f}")
        print(f"  Adj Edge Jaccard: {result['adj_edge_jaccard']:.4f}")
        print(f"  Time: {elapsed:.1f}s")
    
    # Micro-averaged score
    total_tp = totals['edge_tp']
    total_fp = totals['edge_fp']
    total_fn = totals['edge_fn']
    denom = total_tp + total_fp + total_fn
    
    micro_jaccard = total_tp / denom if denom > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  MICRO-AVERAGED RESULTS ({len(per_movie)} movies)")
    print(f"{'='*60}")
    print(f"  Total TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Edge Jaccard (micro): {micro_jaccard:.4f}")
    print(f"  Estimated Kaggle Score: ~{micro_jaccard:.3f}")
    print(f"{'='*60}\n")
    
    return per_movie, micro_jaccard


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--n-movies", type=int, default=5)
    args = parser.parse_args()
    
    # Import our pipeline
    from cell_tracking_notebook import process_dataset
    
    evaluate_on_training(args.train_dir, process_dataset, args.n_movies)
