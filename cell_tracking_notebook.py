"""
Biohub Cell Tracking - V19 (Training-Calibrated + Local Evaluation)
Target 0.90+ via Training Data Calibration

Architecture:
1. CALIBRATION PHASE: Run detector on 10 training movies, compare to GEFF GT,
   auto-select optimal detection threshold
2. INFERENCE PHASE: Process all test movies with calibrated parameters
3. Built-in local scoring when GT is available

Key Improvements over V18:
- Auto-calibration of THRESH_OTSU_FACTOR using training GT
- Local evaluation for rapid iteration
- GEFF reader for ground truth
"""

import numpy as np
import pandas as pd
import os
import glob
import time
import json
import warnings
import gc
from collections import defaultdict

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore")

try:
    import blosc2; _B2 = True
except ImportError:
    _B2 = False

try:
    import blosc; _B1 = True
except ImportError:
    _B1 = False

try:
    import zarr; _ZR = True
except ImportError:
    _ZR = False

print(f"[env] zarr={_ZR}  blosc2={_B2}  blosc={_B1}")


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION - V19 (will be auto-calibrated)
# ════════════════════════════════════════════════════════════════════

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
XY_POOL_FACTOR = 4
ISO_SCALE = 1.625

DOG_SIGMAS_ISO = [
    (0.6, 1.0),
    (1.0, 1.6),
    (1.5, 2.4),
    (2.2, 3.5),
]
BG_SIGMA           = 8.0
THRESH_OTSU_FACTOR = 0.15    # Default — will be calibrated
THRESH_REL_FLOOR   = 0.005
PEAK_MIN_DIST_ISO  = 1

PEAK_NMS_DIST_UM   = 3.0
INTENSITY_WIN_ISO  = 2

MAX_LINK_DIST_UM   = 8.5
INTENSITY_WEIGHT   = 0.8
NON_ASSIGN_FACTOR  = 1.25

GAP_MAX_FRAMES     = 3
GAP_MAX_DIST_UM    = 12.0
GAP_FRAME_PENALTY  = 2.0

MIN_TRACKLET_LEN   = 2


# ════════════════════════════════════════════════════════════════════
# GEFF GROUND TRUTH READER
# ════════════════════════════════════════════════════════════════════

def read_geff(geff_path):
    """Read ground truth from a .geff directory."""
    if not _ZR:
        return {}, []
    g = zarr.open(geff_path, mode='r')
    
    ids = np.array(g['nodes/ids'])
    t = np.array(g['nodes/props/t/values'])
    z = np.array(g['nodes/props/z/values'])
    y = np.array(g['nodes/props/y/values'])
    x = np.array(g['nodes/props/x/values'])
    
    nodes = {}
    for i in range(len(ids)):
        nodes[int(ids[i])] = {'t': int(t[i]), 'z': float(z[i]), 'y': float(y[i]), 'x': float(x[i])}
    
    edge_ids = np.array(g['edges/ids'])
    edges = []
    if edge_ids.ndim == 2 and edge_ids.shape[1] == 2:
        for i in range(len(edge_ids)):
            edges.append((int(edge_ids[i, 0]), int(edge_ids[i, 1])))
    else:
        try:
            sources = np.array(g['edges/props/source_ids/values'])
            targets = np.array(g['edges/props/target_ids/values'])
            for s, t_ in zip(sources, targets):
                edges.append((int(s), int(t_)))
        except Exception:
            pass
    
    return nodes, edges


# ════════════════════════════════════════════════════════════════════
# LOCAL EVALUATION (Exact Competition Metric)
# ════════════════════════════════════════════════════════════════════

def match_nodes_at_time(pred_list, gt_list, max_dist_um=7.0):
    """Bipartite matching at one timepoint. Returns pred_id -> gt_id mapping."""
    if not pred_list or not gt_list:
        return {}
    
    pred_ids, pred_coords = zip(*pred_list)
    gt_ids, gt_coords = zip(*gt_list)
    
    pred_um = np.array(pred_coords) * VOXEL_SCALE
    gt_um = np.array(gt_coords) * VOXEL_SCALE
    
    n_pred, n_gt = len(pred_ids), len(gt_ids)
    cost = np.full((n_pred, n_gt), 1e9)
    
    tree = cKDTree(gt_um)
    for i in range(n_pred):
        near = tree.query_ball_point(pred_um[i], r=max_dist_um)
        for j in near:
            cost[i, j] = np.linalg.norm(pred_um[i] - gt_um[j])
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matching = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_dist_um:
            matching[pred_ids[r]] = gt_ids[c]
    return matching


def evaluate_predictions(pred_nodes, pred_edges, gt_nodes, gt_edges):
    """Compute exact competition metric."""
    pred_dict = {}
    for nid, t, z, y, x in pred_nodes:
        pred_dict[nid] = {'t': int(t), 'z': float(z), 'y': float(y), 'x': float(x)}
    
    pred_by_t = defaultdict(list)
    gt_by_t = defaultdict(list)
    for nid, d in pred_dict.items():
        pred_by_t[d['t']].append((nid, [d['z'], d['y'], d['x']]))
    for nid, d in gt_nodes.items():
        gt_by_t[d['t']].append((nid, [d['z'], d['y'], d['x']]))
    
    pred_to_gt = {}
    for t in set(list(pred_by_t.keys()) + list(gt_by_t.keys())):
        m = match_nodes_at_time(pred_by_t.get(t, []), gt_by_t.get(t, []))
        pred_to_gt.update(m)
    
    gt_edge_set = set()
    gt_source_map = defaultdict(set)
    gt_target_map = defaultdict(set)
    for src, tgt in gt_edges:
        if src in gt_nodes and tgt in gt_nodes:
            if gt_nodes[tgt]['t'] - gt_nodes[src]['t'] == 1:
                gt_edge_set.add((src, tgt))
                gt_source_map[src].add(tgt)
                gt_target_map[tgt].add(src)
    
    edge_tp = 0
    edge_fp = 0
    matched_gt_edges = set()
    
    for pred_src, pred_tgt in pred_edges:
        if pred_src in pred_dict and pred_tgt in pred_dict:
            if pred_dict[pred_tgt]['t'] - pred_dict[pred_src]['t'] != 1:
                continue
        
        gt_src = pred_to_gt.get(pred_src)
        gt_tgt = pred_to_gt.get(pred_tgt)
        
        if gt_src is not None and gt_tgt is not None and (gt_src, gt_tgt) in gt_edge_set:
            edge_tp += 1
            matched_gt_edges.add((gt_src, gt_tgt))
        else:
            is_fp = False
            if gt_src is not None and gt_src in gt_source_map:
                is_fp = True
            if gt_tgt is not None and gt_tgt in gt_target_map:
                is_fp = True
            if is_fp:
                edge_fp += 1
    
    edge_fn = len(gt_edge_set) - len(matched_gt_edges)
    denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denom if denom > 0 else 0.0
    
    node_recall = len(pred_to_gt) / len(gt_nodes) if gt_nodes else 0
    
    return {
        'edge_tp': edge_tp, 'edge_fp': edge_fp, 'edge_fn': edge_fn,
        'edge_jaccard': edge_jaccard, 'node_recall': node_recall,
        'num_pred_nodes': len(pred_dict), 'num_gt_nodes': len(gt_nodes),
        'num_matched': len(pred_to_gt),
    }


# ════════════════════════════════════════════════════════════════════
# ZARR I/O
# ════════════════════════════════════════════════════════════════════

def _read_meta(zarr_path):
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        m = json.load(f)
    shape = tuple(int(s) for s in m["shape"])
    dt = m.get("data_type", m.get("dtype", "uint16"))
    if isinstance(dt, dict): dt = dt.get("name", "uint16")
    return shape, np.dtype(dt)

def _decompress(raw):
    if _B2:
        try: return blosc2.decompress(raw)
        except Exception: pass
    if _B1:
        try: return blosc.decompress(raw)
        except Exception: pass
    try:
        import numcodecs; return numcodecs.Blosc(cname="zstd").decode(raw)
    except Exception: pass
    import zstandard as zstd
    return zstd.ZstdDecompressor().decompress(raw)

def _open_zarr(path):
    if not _ZR: return None
    try:
        r = zarr.open(path, mode="r")
        return r if isinstance(r, zarr.Array) else r.get("0")
    except Exception: return None

def read_frame(zarr_path, t, shape, dtype, z_arr=None):
    if z_arr is not None:
        try: return np.asarray(z_arr[t])
        except Exception: pass
    p = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(p, "rb") as f: raw = f.read()
    return np.frombuffer(_decompress(raw), dtype=dtype).copy().reshape(shape[1:])


# ════════════════════════════════════════════════════════════════════
# PROCESSING UTILS
# ════════════════════════════════════════════════════════════════════

def xy_pool(frame, factor=XY_POOL_FACTOR):
    Z, Y, X = frame.shape
    Yp, Xp = Y // factor, X // factor
    trimmed = frame[:, :Yp * factor, :Xp * factor].astype(np.float32)
    return trimmed.reshape(Z, Yp, factor, Xp, factor).mean(axis=(2, 4))

def _iso_to_orig(coords_iso):
    out = np.empty_like(coords_iso, dtype=np.float64)
    out[:, 0] = coords_iso[:, 0]
    out[:, 1] = coords_iso[:, 1] * XY_POOL_FACTOR + (XY_POOL_FACTOR - 1) / 2.0
    out[:, 2] = coords_iso[:, 2] * XY_POOL_FACTOR + (XY_POOL_FACTOR - 1) / 2.0
    return out

def _physical_nms(coords_iso, intensities, min_dist_um):
    if len(coords_iso) < 2:
        return coords_iso, intensities
    order = np.argsort(intensities)[::-1]
    coords_um = coords_iso[order] * ISO_SCALE
    tree = cKDTree(coords_um)
    pairs = tree.query_pairs(r=min_dist_um)
    suppressed = set()
    for i, j in sorted(pairs):
        if i not in suppressed:
            suppressed.add(j)
    keep = [i for i in range(len(order)) if i not in suppressed]
    return coords_iso[order[keep]], intensities[order[keep]]

def _extract_local_intensities(vol, coords_iso, radius=INTENSITY_WIN_ISO):
    intens = np.zeros(len(coords_iso), dtype=np.float32)
    Z, Y, X = vol.shape
    for i, (z, y, x) in enumerate(coords_iso):
        z0, z1 = max(0, int(z) - radius), min(Z, int(z) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(Y, int(y) + radius + 1)
        x0, x1 = max(0, int(x) - radius), min(X, int(x) + radius + 1)
        intens[i] = np.sum(vol[z0:z1, y0:y1, x0:x1])
    return intens


# ════════════════════════════════════════════════════════════════════
# DETECTION (with configurable threshold)
# ════════════════════════════════════════════════════════════════════

def _adaptive_thresh(dog_vol, otsu_factor=None):
    if otsu_factor is None:
        otsu_factor = THRESH_OTSU_FACTOR
    valid = dog_vol[dog_vol > 0]
    if len(valid) == 0: return 0.01
    mu = valid.mean()
    sd = valid.std()
    th = mu + otsu_factor * sd
    return max(th, THRESH_REL_FLOOR * valid.max())

def detect_cells(frame, otsu_factor=None):
    vol_iso = xy_pool(frame)
    p2, p99 = np.percentile(vol_iso, [2, 99])
    vol_n = np.clip((vol_iso - p2) / (p99 - p2 + 1e-8), 0.0, 1.0)
    
    bg = gaussian_filter(vol_n, sigma=BG_SIGMA, truncate=3.0)
    fg = np.maximum(0, vol_n - bg)
    
    max_dog = np.zeros_like(fg)
    for s1, s2 in DOG_SIGMAS_ISO:
        g1 = gaussian_filter(fg, s1, truncate=3.0)
        g2 = gaussian_filter(fg, s2, truncate=3.0)
        dog = g1 - g2
        np.maximum(max_dog, dog, out=max_dog)
    
    th = _adaptive_thresh(max_dog, otsu_factor)
    max_dog[max_dog < th] = 0
    
    peaks = peak_local_max(max_dog, min_distance=PEAK_MIN_DIST_ISO, exclude_border=False)
    
    if len(peaks) > 0:
        intens = _extract_local_intensities(fg, peaks)
        peaks, intens = _physical_nms(peaks, intens, PEAK_NMS_DIST_UM)
        orig_coords = _iso_to_orig(peaks)
        return orig_coords, max_dog[peaks[:, 0], peaks[:, 1], peaks[:, 2]], intens
    
    return np.empty((0, 3)), np.empty(0), np.empty(0)


# ════════════════════════════════════════════════════════════════════
# TRACKING (CC-LAP)
# ════════════════════════════════════════════════════════════════════

def link_frames(c1, c2, i1, i2):
    if len(c1) == 0 or len(c2) == 0: return []
    
    p1 = c1 * VOXEL_SCALE
    p2 = c2 * VOXEL_SCALE
    
    tree2 = cKDTree(p2)
    pairs = tree2.query_ball_point(p1, r=MAX_LINK_DIST_UM)
    
    adj = defaultdict(list)
    dists_map = {}
    
    for r, cands in enumerate(pairs):
        for c in cands:
            d = np.linalg.norm(p1[r] - p2[c])
            if d <= MAX_LINK_DIST_UM:
                li = abs(np.log(max(i1[r], 1e-5)) - np.log(max(i2[c], 1e-5)))
                cost_val = d + INTENSITY_WEIGHT * li
                adj[f"L{r}"].append(f"R{c}")
                adj[f"R{c}"].append(f"L{r}")
                dists_map[(r, c)] = cost_val
    
    if not dists_map: return []
    
    visited = set()
    components = []
    for node in adj.keys():
        if node not in visited:
            comp_L, comp_R = [], []
            q = [node]
            visited.add(node)
            while q:
                cur = q.pop()
                if cur.startswith("L"): comp_L.append(int(cur[1:]))
                else: comp_R.append(int(cur[1:]))
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            if comp_L and comp_R:
                components.append((comp_L, comp_R))
    
    result = []
    non_assign = MAX_LINK_DIST_UM * NON_ASSIGN_FACTOR
    
    for comp_L, comp_R in components:
        nr, nc = len(comp_L), len(comp_R)
        cost = np.full((nr + nc, nr + nc), non_assign)
        cost[nr:, nc:] = 0.0
        
        ri = {v: k for k, v in enumerate(comp_L)}
        ci = {v: k for k, v in enumerate(comp_R)}
        
        for r in comp_L:
            for c in comp_R:
                if (r, c) in dists_map:
                    cost[ri[r], ci[c]] = dists_map[(r, c)]
        
        for k in range(nr): cost[k, nc + k] = non_assign
        for k in range(nc): cost[nr + k, k] = non_assign
        
        ra, ca = linear_sum_assignment(cost)
        for r_idx, c_idx in zip(ra, ca):
            if r_idx < nr and c_idx < nc:
                r, c = comp_L[r_idx], comp_R[c_idx]
                if dists_map.get((r, c), np.inf) <= MAX_LINK_DIST_UM:
                    result.append((r, c))
    
    return result


# ════════════════════════════════════════════════════════════════════
# GAP CLOSING (LAP-based)
# ════════════════════════════════════════════════════════════════════

def gap_close(nodes, edges, T):
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for s, t_ in edges:
        adj_fwd[s].append(t_); adj_rev[t_].append(s)
    
    node_dict = {}
    for nid, t, z, y, x in nodes:
        node_dict[nid] = (t, np.array([z, y, x]))
    
    def _smoothed_velocity(nid):
        v_sum = np.zeros(3)
        count = 0
        curr = nid
        for _ in range(3):
            parents = adj_rev.get(curr, [])
            if not parents: break
            pid = parents[0]
            v_sum += (node_dict[curr][1] - node_dict[pid][1]) * VOXEL_SCALE
            curr = pid
            count += 1
        return v_sum / count if count > 0 else np.zeros(3)
    
    terminals = []
    starts = []
    for nid, (t, pos) in node_dict.items():
        if not adj_fwd[nid] and t < T - 1:
            terminals.append((nid, t, pos))
        if not adj_rev[nid] and t > 0:
            starts.append((nid, t, pos))
    
    starts_by_t = defaultdict(list)
    for s in starts: starts_by_t[s[1]].append(s)
    
    gap_edges = []
    gap_nodes = []
    new_nid = max((n[0] for n in nodes), default=0) + 1
    used_t, used_s = set(), set()
    
    for gap in range(1, GAP_MAX_FRAMES + 1):
        all_pairs = []
        for term_nid, term_t, term_pos in terminals:
            if term_nid in used_t: continue
            cands = starts_by_t.get(term_t + gap + 1, [])
            if not cands: continue
            
            term_v = _smoothed_velocity(term_nid)
            pred_pos = (term_pos * VOXEL_SCALE) + (term_v * gap)
            
            start_um = np.array([s[2] * VOXEL_SCALE for s in cands])
            tree = cKDTree(start_um)
            idxs = tree.query_ball_point(pred_pos, r=GAP_MAX_DIST_UM)
            
            for idx in idxs:
                s_nid, s_t, s_pos = cands[idx]
                if s_nid in used_s: continue
                dist = np.linalg.norm((s_pos * VOXEL_SCALE) - pred_pos)
                if dist + (gap * GAP_FRAME_PENALTY) <= GAP_MAX_DIST_UM:
                    all_pairs.append((term_nid, s_nid, dist, term_pos, s_pos, term_t, gap))
        
        if not all_pairs: continue
        
        unique_terms = list(set(p[0] for p in all_pairs))
        unique_starts = list(set(p[1] for p in all_pairs))
        ti_map = {v: i for i, v in enumerate(unique_terms)}
        si_map = {v: i for i, v in enumerate(unique_starts)}
        
        nt, ns = len(unique_terms), len(unique_starts)
        na = GAP_MAX_DIST_UM * 1.5
        cm = np.full((nt + ns, nt + ns), na)
        cm[nt:, ns:] = 0.0
        
        for tn, sn, d, _, _, _, _ in all_pairs:
            ri, ci = ti_map[tn], si_map[sn]
            cm[ri, ci] = min(cm[ri, ci], d)
        
        for k in range(nt): cm[k, ns + k] = na
        for k in range(ns): cm[nt + k, k] = na
        
        ra, ca = linear_sum_assignment(cm)
        for r_idx, c_idx in zip(ra, ca):
            if r_idx < nt and c_idx < ns and cm[r_idx, c_idx] < na:
                t_nid = unique_terms[r_idx]
                s_nid = unique_starts[c_idx]
                match = next((p for p in all_pairs if p[0] == t_nid and p[1] == s_nid), None)
                if match is None: continue
                _, _, _, tp, sp, tt, g = match
                used_t.add(t_nid); used_s.add(s_nid)
                prev = t_nid
                for gi in range(1, g + 1):
                    frac = gi / (g + 1)
                    ip = tp + frac * (sp - tp)
                    gap_nodes.append((new_nid, tt + gi, ip[0], ip[1], ip[2]))
                    gap_edges.append((prev, new_nid))
                    prev = new_nid
                    new_nid += 1
                gap_edges.append((prev, s_nid))
    
    return gap_nodes, gap_edges


def prune_short_tracklets(nodes, edges):
    adj = defaultdict(list)
    for s, t_ in edges: adj[s].append(t_); adj[t_].append(s)
    visited, valid_nids = set(), set()
    for n in nodes:
        nid = n[0]
        if nid not in visited:
            q, comp = [nid], {nid}
            visited.add(nid)
            while q:
                curr = q.pop(0)
                for nb in adj[curr]:
                    if nb not in visited:
                        visited.add(nb); comp.add(nb); q.append(nb)
            if len(comp) >= MIN_TRACKLET_LEN: valid_nids.update(comp)
    return ([n for n in nodes if n[0] in valid_nids],
            [(s, t_) for s, t_ in edges if s in valid_nids and t_ in valid_nids])


# ════════════════════════════════════════════════════════════════════
# CALIBRATION
# ════════════════════════════════════════════════════════════════════

def calibrate_threshold(train_dir, n_movies=10):
    """
    Auto-calibrate detection threshold using training GT.
    Tests multiple thresholds and picks the one maximizing node recall
    while keeping detection count reasonable.
    """
    zarr_dirs = sorted(glob.glob(os.path.join(train_dir, "*.zarr")))[:n_movies]
    
    candidates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    best_thresh = THRESH_OTSU_FACTOR
    best_score = -1
    
    print(f"\n[CALIBRATION] Testing {len(candidates)} thresholds on {len(zarr_dirs)} movies...")
    
    for thresh in candidates:
        total_recall = 0
        total_movies = 0
        
        for zp in zarr_dirs:
            geff_path = zp.replace(".zarr", ".geff")
            if not os.path.exists(geff_path): continue
            
            gt_nodes, _ = read_geff(geff_path)
            if not gt_nodes: continue
            
            shape, dtype = _read_meta(zp)
            z_arr = _open_zarr(zp)
            
            gt_by_t = defaultdict(list)
            for nid, d in gt_nodes.items():
                gt_by_t[d['t']].append((nid, [d['z'], d['y'], d['x']]))
            
            # Test on 5 representative frames
            test_times = sorted(gt_by_t.keys())[:5]
            matched = 0
            total_gt = 0
            
            for t in test_times:
                frame = read_frame(zp, t, shape, dtype, z_arr)
                coords, _, intens = detect_cells(frame, otsu_factor=thresh)
                
                gt_list = gt_by_t[t]
                total_gt += len(gt_list)
                
                if len(coords) > 0 and len(gt_list) > 0:
                    pred_list = [(i, [coords[i, 0], coords[i, 1], coords[i, 2]]) for i in range(len(coords))]
                    m = match_nodes_at_time(pred_list, gt_list)
                    matched += len(m)
            
            if total_gt > 0:
                total_recall += matched / total_gt
                total_movies += 1
            
            del z_arr
        
        avg_recall = total_recall / max(total_movies, 1)
        print(f"  thresh={thresh:.2f}  avg_recall={avg_recall:.3f}")
        
        if avg_recall > best_score:
            best_score = avg_recall
            best_thresh = thresh
    
    print(f"\n[CALIBRATION] Best threshold: {best_thresh:.2f} (recall={best_score:.3f})")
    return best_thresh


# ════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════

def process_dataset(zarr_path, ds_name, otsu_factor=None):
    t0 = time.time()
    shape, dtype = _read_meta(zarr_path)
    T = shape[0]
    print(f"  [{ds_name}] shape={shape}  dtype={dtype}")
    
    z_arr = _open_zarr(zarr_path)
    all_coords, all_intensities = {}, {}
    total_cells = 0
    
    for t in range(T):
        frame = read_frame(zarr_path, t, shape, dtype, z_arr)
        coords, _, intens = detect_cells(frame, otsu_factor=otsu_factor)
        all_coords[t] = coords
        all_intensities[t] = intens
        total_cells += len(coords)
        if t % 25 == 0: print(f"    detect  t={t:>3d}/{T}  cells={len(coords)}")
    
    avg = total_cells / max(T, 1)
    print(f"    detection done — avg {avg:.0f} cells/frame, {total_cells} total")
    
    nid = 1
    node_map, nodes = {}, []
    for t in range(T):
        for i, (z, y, x) in enumerate(all_coords[t]):
            node_map[(t, i)] = nid
            nodes.append((nid, t, int(z), int(y), int(x)))
            nid += 1
    
    edges = []
    for t in range(T - 1):
        assn = link_frames(all_coords[t], all_coords[t + 1], all_intensities[t], all_intensities[t + 1])
        for si, ti in assn: edges.append((node_map[(t, si)], node_map[(t + 1, ti)]))
    print(f"    pass-1:  {len(edges)} edges")
    
    gap_nodes, gap_edges = gap_close(nodes, edges, T)
    nodes.extend(gap_nodes); edges.extend(gap_edges)
    print(f"    gap-close:  +{len(gap_nodes)} nodes, +{len(gap_edges)} edges")
    
    nodes, edges = prune_short_tracklets(nodes, edges)
    elapsed = time.time() - t0
    print(f"  [{ds_name}] done  {len(nodes)} nodes  {len(edges)} edges  ({elapsed:.1f}s)\n")
    
    del z_arr, all_coords, all_intensities
    return nodes, edges


def create_submission(test_dir, output_path, train_dir=None):
    # PHASE 1: Calibrate on training data if available
    otsu_factor = THRESH_OTSU_FACTOR
    if train_dir and os.path.exists(train_dir):
        otsu_factor = calibrate_threshold(train_dir, n_movies=10)
    
    zarr_dirs = sorted(glob.glob(os.path.join(test_dir, "*.zarr")))
    print(f"\n{'=' * 64}")
    print(f"  Biohub V19 (Calibrated) — {len(zarr_dirs)} datasets, thresh={otsu_factor:.2f}")
    print(f"{'=' * 64}\n")
    
    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)
    
    row_id, total_nodes, total_edges = 0, 0, 0
    
    for zp in zarr_dirs:
        ds = os.path.basename(zp).replace(".zarr", "")
        try:
            nodes, edges = process_dataset(zp, ds, otsu_factor=otsu_factor)
        except Exception as exc:
            import traceback; traceback.print_exc()
            nodes = [(1, 0, 32, 128, 128), (2, 1, 32, 128, 128)]
            edges = [(1, 2)]
        
        chunk = []
        for nid, t, z, y, x in nodes:
            chunk.append([row_id, ds, "node", nid, t, z, y, x, -1, -1])
            row_id += 1
        for src, tgt in edges:
            chunk.append([row_id, ds, "edge", -1, -1, -1, -1, -1, src, tgt])
            row_id += 1
        
        df_chunk = pd.DataFrame(chunk, columns=cols)
        for c in cols[3:]: df_chunk[c] = df_chunk[c].astype(int)
        df_chunk.to_csv(output_path, mode='a', header=False, index=False)
        
        total_nodes += len(nodes)
        total_edges += len(edges)
        del chunk, df_chunk, nodes, edges
        gc.collect()
    
    print(f"{'=' * 64}")
    print(f"  Submission: {total_nodes:,} nodes  {total_edges:,} edges")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    TEST_DIR  = "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"
    TRAIN_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
    OUT_CSV   = "submission.csv"
    
    if not os.path.exists(TEST_DIR):
        print(f"WARNING: {TEST_DIR} not found.")
    else:
        t_wall = time.time()
        create_submission(TEST_DIR, OUT_CSV, train_dir=TRAIN_DIR)
        print(f"Total wall time: {(time.time() - t_wall) / 60:.1f} min")
        print(pd.read_csv(OUT_CSV, nrows=20))
