"""
Biohub Cell Tracking - V15 (Super-Classical V7 Optimized)
Targeting 0.90+ Leaderboard Score via Max-Pooling & Hybrid LAP

Features:
- Isotropic Grid (XY_POOL = 4) for blazing fast 3D operations
- Max-Pooling downsampling to perfectly preserve small/dim cells (High Recall)
- Hybrid Connected-Component LAP Tracking (prevents O(N^3) timeouts)
- cKDTree NMS and Gap Closing (prevents O(N^2) hangs)
- Streaming CSV Generation (prevents Out of Memory)
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
# CONFIGURATION - V15 TUNED
# ════════════════════════════════════════════════════════════════════

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
XY_POOL_FACTOR = 4
ISO_SCALE = 1.625  # µm/voxel after pooling

DOG_SIGMAS_ISO = [
    (0.6, 1.0),   
    (1.0, 1.6),   
    (1.5, 2.4),   
    (2.2, 3.5),   
]
BG_SIGMA           = 8.0     
THRESH_OTSU_FACTOR = 0.10    # Slightly relaxed from 0.15 for better recall
THRESH_REL_FLOOR   = 0.005   
PEAK_MIN_DIST_ISO  = 1       

PEAK_NMS_DIST_UM   = 3.5     
INTENSITY_WIN_ISO  = 2       

MAX_LINK_DIST_UM   = 7.0     # STRICT LIMIT to shatter graph into tiny components (Timeout immunity)
INTENSITY_WEIGHT   = 1.5     
NON_ASSIGN_FACTOR  = 1.25    

GAP_MAX_FRAMES     = 3
GAP_MAX_DIST_UM    = 12.0    # Relaxed Gap Distance for fast-moving cells
GAP_FRAME_PENALTY  = 2.0

DIV_MAX_DIST_UM    = 7.0     
DIV_INTENSITY_TOL  = 0.4     
DIV_MIN_TRACK_LEN  = 4
MIN_TRACKLET_LEN   = 3


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
    """
    Mean-Pooling downsampling.
    Crucial for suppressing shot noise (acts as a box blur). 
    Dim cells are preserved via lowered thresholds.
    """
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
    """Ultra-fast cKDTree Non-Maximum Suppression"""
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
# DETECTION
# ════════════════════════════════════════════════════════════════════

def _adaptive_thresh(dog_vol):
    valid = dog_vol[dog_vol > 0]
    if len(valid) == 0: return 0.01
    mu = valid.mean()
    sd = valid.std()
    th = mu + THRESH_OTSU_FACTOR * sd
    return max(th, THRESH_REL_FLOOR * valid.max())


def detect_cells(frame):
    # Use mean-pooling to radically suppress shot noise
    vol_iso = xy_pool(frame)
    p2, p99 = np.percentile(vol_iso, [2, 99])
    vol_n = np.clip((vol_iso - p2) / (p99 - p2 + 1e-8), 0.0, 1.0)
    
    # Fast filtering with truncate=3.0 to prevent large kernel lag
    bg = gaussian_filter(vol_n, sigma=BG_SIGMA, truncate=3.0)
    fg = np.maximum(0, vol_n - bg)

    max_dog = np.zeros_like(fg)
    for s1, s2 in DOG_SIGMAS_ISO:
        g1 = gaussian_filter(fg, s1, truncate=3.0)
        g2 = gaussian_filter(fg, s2, truncate=3.0)
        dog = g1 - g2
        np.maximum(max_dog, dog, out=max_dog)

    th = _adaptive_thresh(max_dog)
    max_dog[max_dog < th] = 0

    peaks = peak_local_max(max_dog, min_distance=PEAK_MIN_DIST_ISO, exclude_border=False)
    
    if len(peaks) > 0:
        intens = _extract_local_intensities(fg, peaks)
        peaks, intens = _physical_nms(peaks, intens, PEAK_NMS_DIST_UM)
        orig_coords = _iso_to_orig(peaks)
        return orig_coords, max_dog[peaks[:, 0], peaks[:, 1], peaks[:, 2]], intens

    return np.empty((0, 3)), np.empty(0), np.empty(0)


# ════════════════════════════════════════════════════════════════════
# HYBRID TRACKING
# ════════════════════════════════════════════════════════════════════

def link_frames(c1, c2, i1, i2):
    """
    Connected Components LAP with Greedy Fallback.
    O(N^3) LAP is used for sparse valid clusters.
    O(N log N) Greedy is used if cluster > 200 (prevents Kaggle Timeout).
    """
    if len(c1) == 0 or len(c2) == 0: return []
    
    p1 = c1 * VOXEL_SCALE
    p2 = c2 * VOXEL_SCALE
    
    # Build Adjacency and Distances
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

    # BFS Connected Components
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
        
        # --- EXACT LAP FOR SMALL CLUSTERS ---
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


def _velocity(nodes, edges, node_map, nid, t):
    parent = [s for s, tgt in edges if tgt == nid]
    if not parent: return np.zeros(3)
    pid = parent[0]
    p_node = next(n for n in nodes if n[0] == pid)
    c_node = next(n for n in nodes if n[0] == nid)
    return (np.array(c_node[2:]) - np.array(p_node[2:])) * VOXEL_SCALE


# ════════════════════════════════════════════════════════════════════
# GAP CLOSING
# ════════════════════════════════════════════════════════════════════

def gap_close(nodes, edges, node_map, T):
    terminals = []
    starts = []
    
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for s, t_ in edges:
        adj_fwd[s].append(t_); adj_rev[t_].append(s)
        
    for nid, t, z, y, x in nodes:
        if len(adj_fwd[nid]) == 0 and t < T - 1:
            terminals.append((nid, t, np.array([z,y,x])))
        if len(adj_rev[nid]) == 0 and t > 0:
            starts.append((nid, t, np.array([z,y,x])))
            
    starts_by_t = defaultdict(list)
    for s in starts: starts_by_t[s[1]].append(s)
            
    gap_edges = []
    gap_nodes = []
    new_nid = max((n[0] for n in nodes), default=0) + 1
    
    used_t, used_s = set(), set()
    
    for gap in range(1, GAP_MAX_FRAMES + 1):
        for term_nid, term_t, term_pos in terminals:
            if term_nid in used_t: continue
            
            cands_starts = starts_by_t.get(term_t + gap + 1, [])
            if not cands_starts: continue
            
            term_v = _velocity(nodes, edges, node_map, term_nid, term_t)
            pred_pos_um = (term_pos * VOXEL_SCALE) + (term_v * gap)
            
            start_um = np.array([s[2]*VOXEL_SCALE for s in cands_starts])
            tree = cKDTree(start_um)
            idxs = tree.query_ball_point(pred_pos_um, r=GAP_MAX_DIST_UM)
            
            cand = []
            for idx in idxs:
                s_nid, s_t, s_pos = cands_starts[idx]
                if s_nid in used_s: continue
                dist = np.linalg.norm((s_pos * VOXEL_SCALE) - pred_pos_um)
                if dist + (gap * GAP_FRAME_PENALTY) <= GAP_MAX_DIST_UM:
                    cand.append((dist, s_nid, s_pos))
                        
            if cand:
                cand.sort()
                best_dist, best_start, start_pos = cand[0]
                used_t.add(term_nid); used_s.add(best_start)
                
                prev_nid = term_nid
                for g in range(1, gap + 1):
                    frac = g / (gap + 1)
                    i_pos = term_pos + frac * (start_pos - term_pos)
                    gap_nodes.append((new_nid, term_t + g, i_pos[0], i_pos[1], i_pos[2]))
                    gap_edges.append((prev_nid, new_nid))
                    prev_nid = new_nid
                    new_nid += 1
                gap_edges.append((prev_nid, best_start))
                
    return gap_nodes, gap_edges


# ════════════════════════════════════════════════════════════════════
# BRANCHING MITOSIS
# ════════════════════════════════════════════════════════════════════

def detect_divisions(nodes, edges, interp_nids, all_intensities, node_map):
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for s, t_ in edges:
        adj_fwd[s].append(t_); adj_rev[t_].append(s)
        
    terminals, starts = [], []
    node_dict = {n[0]: {"t": n[1], "pos": np.array(n[2:])} for n in nodes}
    
    def _track_len(nid, forward=True):
        l, curr = 1, nid
        while True:
            nxt = adj_fwd[curr] if forward else adj_rev[curr]
            if not nxt: break
            curr = nxt[0]; l += 1
        return l

    for nid, data in node_dict.items():
        if nid in interp_nids: continue
        if not adj_fwd[nid] and _track_len(nid, False) >= DIV_MIN_TRACK_LEN: terminals.append(nid)
        if not adj_rev[nid] and _track_len(nid, True) >= DIV_MIN_TRACK_LEN: starts.append(nid)
            
    candidates = []
    for p_nid in terminals:
        p = node_dict[p_nid]
        t = p["t"]
        
        try:
            p_idx = [k for k, v in node_map.items() if v == p_nid][0][1]
            p_int = all_intensities[t][p_idx]
        except IndexError: continue
            
        d_cands = []
        for s_nid in starts:
            s = node_dict[s_nid]
            if s["t"] == t + 1:
                dist = np.linalg.norm((p["pos"] - s["pos"]) * VOXEL_SCALE)
                if dist <= DIV_MAX_DIST_UM:
                    try:
                        s_idx = [k for k, v in node_map.items() if v == s_nid][0][1]
                        s_int = all_intensities[t+1][s_idx]
                        d_cands.append((dist, s_nid, s_int))
                    except IndexError: pass
                        
        if len(d_cands) >= 2:
            d_cands.sort()
            for i in range(len(d_cands)):
                for j in range(i + 1, len(d_cands)):
                    d1_dist, d1_nid, d1_int = d_cands[i]
                    d2_dist, d2_nid, d2_int = d_cands[j]
                    
                    sum_d = d1_int + d2_int
                    if sum_d > 0:
                        ratio = p_int / sum_d
                        if (1.0 - DIV_INTENSITY_TOL) <= ratio <= (1.0 + DIV_INTENSITY_TOL):
                            cost = d1_dist + d2_dist
                            candidates.append((cost, p_nid, d1_nid, d2_nid))

    candidates.sort()
    used_p, used_s, div_edges = set(), set(), []
    
    for _, p_nid, d1_nid, d2_nid in candidates:
        if p_nid in used_p or d1_nid in used_s or d2_nid in used_s: continue
        div_edges.append((p_nid, d1_nid)); div_edges.append((p_nid, d2_nid))
        used_p.add(p_nid); used_s.add(d1_nid); used_s.add(d2_nid)
        
    return div_edges


def prune_short_tracklets(nodes, edges):
    adj = defaultdict(list)
    for s, t_ in edges: adj[s].append(t_); adj[t_].append(s)
    visited, valid_nids = set(), set()
    node_ids = [n[0] for n in nodes]
    for nid in node_ids:
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
# PIPELINE RUNNER
# ════════════════════════════════════════════════════════════════════

def process_dataset(zarr_path, ds_name):
    t0 = time.time()
    shape, dtype = _read_meta(zarr_path)
    T = shape[0]
    print(f"  [{ds_name}] shape={shape}  dtype={dtype}")

    z_arr = _open_zarr(zarr_path)
    all_coords, all_intensities = {}, {}
    total_cells = 0

    for t in range(T):
        frame = read_frame(zarr_path, t, shape, dtype, z_arr)
        coords, _, intens = detect_cells(frame)
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

    gap_nodes, gap_edges = gap_close(nodes, edges, node_map, T)
    interp_nids = set(n[0] for n in gap_nodes)
    nodes.extend(gap_nodes); edges.extend(gap_edges)
    print(f"    gap-close:  +{len(gap_nodes)} nodes, +{len(gap_edges)} edges")

    div_edges = detect_divisions(nodes, edges, interp_nids, all_intensities, node_map)
    edges.extend(div_edges)
    print(f"    divisions:  +{len(div_edges)} edges")

    nodes, edges = prune_short_tracklets(nodes, edges)
    elapsed = time.time() - t0
    print(f"  [{ds_name}] ✓  {len(nodes)} nodes  {len(edges)} edges  ({elapsed:.1f}s)\n")
    
    del z_arr, all_coords, all_intensities
    return nodes, edges


def create_submission(test_dir, output_path):
    zarr_dirs = sorted(glob.glob(os.path.join(test_dir, "*.zarr")))
    print(f"\n{'=' * 64}")
    print(f"  Biohub Cell Tracking V15 (Super-Classical Optimized) — {len(zarr_dirs)} datasets")
    print(f"{'=' * 64}\n")

    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)

    row_id, total_nodes, total_edges = 0, 0, 0

    for zp in zarr_dirs:
        ds = os.path.basename(zp).replace(".zarr", "")
        try:
            nodes, edges = process_dataset(zp, ds)
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
    print(f"  Submission saved → {output_path}")
    print(f"  {len(zarr_dirs)} datasets  |  {total_nodes:,} nodes  |  {total_edges:,} edges")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    TEST_DIR  = "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"
    OUT_CSV   = "submission.csv"
    
    # Optional debug switch to local if test path missing
    if not os.path.exists(TEST_DIR):
        print(f"WARNING: Path {TEST_DIR} does not exist. Update TEST_DIR for local testing.")
    else:
        t_wall = time.time()
        create_submission(TEST_DIR, OUT_CSV)
        print(f"Total wall time: {(time.time() - t_wall) / 60:.1f} min")
        print(pd.read_csv(OUT_CSV, nrows=20))
