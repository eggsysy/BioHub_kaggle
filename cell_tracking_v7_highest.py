"""
Biohub Cell Tracking - V7 Super-Classical Pipeline
Optimized for 0.88+ Leaderboard Score via Biological Priors
- High Recall DoG Detection
- Strict 3D Physical NMS
- Intensity-Conservation Tracking
- Offline Compatible (No Training)
"""

import numpy as np
import pandas as pd
import os
import glob
import time
import json
import warnings
from collections import defaultdict

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore")

# Try to use zarr and blosc if available in environment, else fallback
try:
    import blosc2
    _B2 = True
except ImportError:
    _B2 = False

try:
    import blosc
    _B1 = True
except ImportError:
    _B1 = False

try:
    import zarr
    _ZR = True
except ImportError:
    _ZR = False

print(f"[env] zarr={_ZR}  blosc2={_B2}  blosc={_B1}")


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION - TUNED FOR 0.88+ RECALL & PRECISION
# ════════════════════════════════════════════════════════════════════

# Physical voxel scale (z, y, x  µm/voxel)
VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
XY_POOL_FACTOR = 4
ISO_SCALE = 1.625  # µm/voxel after pooling

# ── Detection (Isotropic DoG) ──
# We use more scales to aggressively catch all cells (high recall)
DOG_SIGMAS_ISO = [
    (0.6, 1.0),   # Tiny fragments/nuclei
    (1.0, 1.6),   # Small nuclei
    (1.5, 2.4),   # Medium nuclei
    (2.2, 3.5),   # Large nuclei
]
BG_SIGMA           = 8.0     # Background subtraction radius
THRESH_OTSU_FACTOR = 0.15    # Lowered to drastically increase recall
THRESH_REL_FLOOR   = 0.005   # Lowered floor for dim cells
PEAK_MIN_DIST_ISO  = 1       # Initial local maxima radius

# ── Physical Non-Maximum Suppression (NMS) ──
# Because recall is high, we must aggressively prune false positives
PEAK_NMS_DIST_UM   = 3.5     # Absolute physical minimum distance between cells
INTENSITY_WIN_ISO  = 2       # Radius to sum intensity around peak

# ── Pass-1 Tracking (Hungarian LAP) ──
MAX_LINK_DIST_UM   = 10.0    # Relaxed to catch fast cells
INTENSITY_WEIGHT   = 1.5     # Weight of log-intensity diff vs physical distance
NON_ASSIGN_FACTOR  = 1.25    # Threshold for creating new track vs linking

# ── Gap Closing ──
GAP_MAX_FRAMES     = 3
GAP_MAX_DIST_UM    = 8.0
GAP_FRAME_PENALTY  = 2.0

# ── Division Detection ──
DIV_MAX_DIST_UM    = 6.0
DIV_INTENSITY_TOL  = 0.4     # Parent must be approx (D1 + D2) ± 40%
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
    if isinstance(dt, dict):
        dt = dt.get("name", "uint16")
    return shape, np.dtype(dt)


def _decompress(raw):
    if _B2:
        try: return blosc2.decompress(raw)
        except Exception: pass
    if _B1:
        try: return blosc.decompress(raw)
        except Exception: pass
    try:
        import numcodecs
        return numcodecs.Blosc(cname="zstd").decode(raw)
    except Exception:
        pass
    import zstandard as zstd
    return zstd.ZstdDecompressor().decompress(raw)


def _open_zarr(path):
    if not _ZR:
        return None
    try:
        r = zarr.open(path, mode="r")
        return r if isinstance(r, zarr.Array) else r.get("0")
    except Exception:
        return None


def read_frame(zarr_path, t, shape, dtype, z_arr=None):
    if z_arr is not None:
        try: return np.asarray(z_arr[t])
        except Exception: pass
    p = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(p, "rb") as f:
        raw = f.read()
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
    """Prune peaks that are too close to brighter peaks."""
    if len(coords_iso) < 2:
        return coords_iso, intensities

    # Sort by intensity descending (brightest first)
    order = np.argsort(intensities)[::-1]
    coords = coords_iso[order]
    intens = intensities[order]

    keep = []
    # Physical distance requires converting to µm
    coords_um = coords * ISO_SCALE
    min_sq = min_dist_um ** 2

    for i in range(len(coords)):
        c_um = coords_um[i]
        conflict = False
        for k in keep:
            k_um = coords_um[k]
            d_sq = np.sum((c_um - k_um) ** 2)
            if d_sq < min_sq:
                conflict = True
                break
        if not conflict:
            keep.append(i)

    keep_idx = order[keep]
    return coords_iso[keep_idx], intensities[keep_idx]


def _extract_local_intensities(vol, coords_iso, radius=INTENSITY_WIN_ISO):
    """Extract sum of intensity in a local window for volume conservation."""
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
    if len(valid) == 0:
        return 0.01
    mu = valid.mean()
    sd = valid.std()
    # Hybrid Otsu-like: mean + fractional standard deviation
    th = mu + THRESH_OTSU_FACTOR * sd
    return max(th, THRESH_REL_FLOOR * valid.max())


def detect_cells(frame):
    vol_iso = xy_pool(frame)
    p2, p99 = np.percentile(vol_iso, [2, 99])
    vol_n = np.clip((vol_iso - p2) / (p99 - p2 + 1e-8), 0.0, 1.0)
    
    bg = gaussian_filter(vol_n, sigma=BG_SIGMA)
    fg = np.maximum(0, vol_n - bg)

    max_dog = np.zeros_like(fg)
    for s1, s2 in DOG_SIGMAS_ISO:
        g1 = gaussian_filter(fg, s1)
        g2 = gaussian_filter(fg, s2)
        dog = g1 - g2
        np.maximum(max_dog, dog, out=max_dog)

    th = _adaptive_thresh(max_dog)
    max_dog[max_dog < th] = 0

    peaks = peak_local_max(max_dog, min_distance=PEAK_MIN_DIST_ISO, exclude_border=False)
    
    if len(peaks) > 0:
        # Extract total brightness of the cell (from bg-subtracted volume)
        intens = _extract_local_intensities(fg, peaks)
        
        # 3D Physical NMS
        peaks, intens = _physical_nms(peaks, intens, PEAK_NMS_DIST_UM)
        
        # Convert to original unpooled coordinates
        orig_coords = _iso_to_orig(peaks)
        return orig_coords, max_dog[peaks[:, 0], peaks[:, 1], peaks[:, 2]], intens

    return np.empty((0, 3)), np.empty(0), np.empty(0)


# ════════════════════════════════════════════════════════════════════
# TRACKING
# ════════════════════════════════════════════════════════════════════

def _cost_matrix(c1, c2, i1, i2):
    """Cost = physical distance + intensity difference penalty"""
    # Physical distance in µm
    p1 = c1 * VOXEL_SCALE
    p2 = c2 * VOXEL_SCALE
    dist_mat = cdist(p1, p2, metric='euclidean')
    
    # Intensity penalty: abs(log(i1 / i2)) -> 0 if equal, high if different
    # This prevents linking a bright cell to a dim cell if they pass each other
    log_i1 = np.log(np.maximum(i1, 1e-5))[:, None]
    log_i2 = np.log(np.maximum(i2, 1e-5))[None, :]
    int_penalty = np.abs(log_i1 - log_i2)
    
    cost_mat = dist_mat + INTENSITY_WEIGHT * int_penalty
    return cost_mat, dist_mat


def link_frames(c1, c2, i1, i2):
    if len(c1) == 0 or len(c2) == 0:
        return [], []
    
    cost_mat, dist_mat = _cost_matrix(c1, c2, i1, i2)
    
    n1, n2 = len(c1), len(c2)
    max_c = cost_mat.max() + 10.0
    
    pad_cost = np.full((n1 + n2, n1 + n2), max_c)
    pad_cost[:n1, :n2] = cost_mat
    
    # Threshold for assignment depends on MAX_LINK_DIST
    non_assign = MAX_LINK_DIST_UM * NON_ASSIGN_FACTOR
    
    # Cost for target to be born (unassigned from c1)
    for j in range(n2):
        pad_cost[n1 + j, j] = non_assign
        
    # Cost for source to die (unassigned to c2)
    for i in range(n1):
        pad_cost[i, n2 + i] = non_assign
        
    # Dummy to Dummy
    pad_cost[n1:, n2:] = 0.0
    
    r_idx, c_idx = linear_sum_assignment(pad_cost)
    
    assignments = []
    unassigned_src = []
    
    for r, c in zip(r_idx, c_idx):
        if r < n1 and c < n2:
            # Physical gate (ignore intensity penalty for hard cutoff)
            if dist_mat[r, c] <= MAX_LINK_DIST_UM:
                assignments.append((r, c))
            else:
                unassigned_src.append(r)
        elif r < n1 and c >= n2:
            unassigned_src.append(r)
            
    return assignments, unassigned_src


def _velocity(nodes, edges, node_map, nid, t):
    parent = [s for s, tgt in edges if tgt == nid]
    if not parent:
        return np.zeros(3)
    pid = parent[0]
    p_node = next(n for n in nodes if n[0] == pid)
    c_node = next(n for n in nodes if n[0] == nid)
    # physical displacement per frame
    v = (np.array(c_node[2:]) - np.array(p_node[2:])) * VOXEL_SCALE
    return v


def gap_close(nodes, edges, all_coords, all_intensities, node_map, T):
    terminals = []
    starts = []
    
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for s, t_ in edges:
        adj_fwd[s].append(t_)
        adj_rev[t_].append(s)
        
    for nid, t, z, y, x in nodes:
        if len(adj_fwd[nid]) == 0 and t < T - 1:
            terminals.append((nid, t, np.array([z,y,x])))
        if len(adj_rev[nid]) == 0 and t > 0:
            starts.append((nid, t, np.array([z,y,x])))
            
    gap_edges = []
    gap_nodes = []
    new_nid = max((n[0] for n in nodes), default=0) + 1
    
    used_t = set()
    used_s = set()
    
    # Sort by gap size (1 frame, then 2 frames...)
    for gap in range(1, GAP_MAX_FRAMES + 1):
        for term_nid, term_t, term_pos in terminals:
            if term_nid in used_t: continue
            
            cand = []
            term_v = _velocity(nodes, edges, node_map, term_nid, term_t)
            # Predict position using constant velocity
            pred_pos_um = (term_pos * VOXEL_SCALE) + (term_v * gap)
            
            for start_nid, start_t, start_pos in starts:
                if start_nid in used_s: continue
                if start_t == term_t + gap + 1:
                    start_um = start_pos * VOXEL_SCALE
                    dist = np.linalg.norm(start_um - pred_pos_um)
                    # Add penalty for larger gaps
                    if dist + (gap * GAP_FRAME_PENALTY) <= GAP_MAX_DIST_UM:
                        cand.append((dist, start_nid, start_pos))
                        
            if cand:
                cand.sort()
                best_dist, best_start, start_pos = cand[0]
                used_t.add(term_nid)
                used_s.add(best_start)
                
                # Interpolate nodes in between
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


def detect_divisions(nodes, edges, interp_nids, all_intensities, node_map):
    """
    Find divisions by linking a track ending (parent) to two starting tracks (daughters).
    Enforces Volume Conservation: I_parent ≈ I_d1 + I_d2.
    """
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for s, t_ in edges:
        adj_fwd[s].append(t_)
        adj_rev[t_].append(s)
        
    terminals = []
    starts = []
    
    node_dict = {n[0]: {"t": n[1], "pos": np.array(n[2:])} for n in nodes}
    
    def _track_len(nid, forward=True):
        l = 1
        curr = nid
        while True:
            nxt = adj_fwd[curr] if forward else adj_rev[curr]
            if not nxt: break
            curr = nxt[0]
            l += 1
        return l

    for nid, data in node_dict.items():
        if nid in interp_nids: continue
        if len(adj_fwd[nid]) == 0 and _track_len(nid, False) >= DIV_MIN_TRACK_LEN:
            terminals.append(nid)
        if len(adj_rev[nid]) == 0 and _track_len(nid, True) >= DIV_MIN_TRACK_LEN:
            starts.append(nid)
            
    candidates = []
    for p_nid in terminals:
        p = node_dict[p_nid]
        t = p["t"]
        
        # Parent intensity
        try:
            p_idx = [k for k, v in node_map.items() if v == p_nid][0][1]
            p_int = all_intensities[t][p_idx]
        except IndexError:
            continue
            
        # Find potential daughters in t+1
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
                    except IndexError:
                        pass
                        
        # Need exactly two valid daughters
        if len(d_cands) >= 2:
            d_cands.sort() # By distance
            for i in range(len(d_cands)):
                for j in range(i + 1, len(d_cands)):
                    d1_dist, d1_nid, d1_int = d_cands[i]
                    d2_dist, d2_nid, d2_int = d_cands[j]
                    
                    # Volume conservation check: Parent ≈ D1 + D2
                    sum_d = d1_int + d2_int
                    if sum_d > 0:
                        ratio = p_int / sum_d
                        if (1.0 - DIV_INTENSITY_TOL) <= ratio <= (1.0 + DIV_INTENSITY_TOL):
                            # Distance metric
                            cost = d1_dist + d2_dist
                            candidates.append((cost, p_nid, d1_nid, d2_nid))

    candidates.sort()
    used_p, used_s = set(), set()
    div_edges = []
    
    for _, p_nid, d1_nid, d2_nid in candidates:
        if p_nid in used_p or d1_nid in used_s or d2_nid in used_s:
            continue
        div_edges.append((p_nid, d1_nid))
        div_edges.append((p_nid, d2_nid))
        used_p.add(p_nid)
        used_s.add(d1_nid)
        used_s.add(d2_nid)
        
    return div_edges


def prune_short_tracklets(nodes, edges):
    adj = defaultdict(list)
    for s, t_ in edges:
        adj[s].append(t_)
        adj[t_].append(s)
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
                        visited.add(nb)
                        comp.add(nb)
                        q.append(nb)
            if len(comp) >= MIN_TRACKLET_LEN:
                valid_nids.update(comp)
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
        if t % 25 == 0:
            print(f"    detect  t={t:>3d}/{T}  cells={len(coords)}")

    avg = total_cells / max(T, 1)
    print(f"    detection done — avg {avg:.0f} cells/frame, {total_cells} total")

    nid = 1
    node_map = {}
    nodes = []
    for t in range(T):
        for i, (z, y, x) in enumerate(all_coords[t]):
            node_map[(t, i)] = nid
            nodes.append((nid, t, int(z), int(y), int(x)))
            nid += 1

    edges = []
    for t in range(T - 1):
        c1, c2 = all_coords[t], all_coords[t + 1]
        i1, i2 = all_intensities[t], all_intensities[t + 1]
        assn, _ = link_frames(c1, c2, i1, i2)
        for si, ti in assn:
            edges.append((node_map[(t, si)], node_map[(t + 1, ti)]))
    print(f"    pass-1:  {len(edges)} edges")

    gap_nodes, gap_edges = gap_close(nodes, edges, all_coords, all_intensities, node_map, T)
    interp_nids = set(n[0] for n in gap_nodes)
    nodes.extend(gap_nodes)
    edges.extend(gap_edges)
    print(f"    gap-close:  +{len(gap_nodes)} nodes, +{len(gap_edges)} edges")

    div_edges = detect_divisions(nodes, edges, interp_nids, all_intensities, node_map)
    edges.extend(div_edges)
    print(f"    divisions:  +{len(div_edges)} edges")

    nodes, edges = prune_short_tracklets(nodes, edges)

    elapsed = time.time() - t0
    print(f"  [{ds_name}] ✓  {len(nodes)} nodes  {len(edges)} edges  ({elapsed:.1f}s)\n")
    return nodes, edges


def create_submission(test_dir, output_path):
    zarr_dirs = sorted(glob.glob(os.path.join(test_dir, "*.zarr")))
    print(f"\n{'=' * 64}")
    print(f"  Biohub Cell Tracking V7 (Super-Classical) — {len(zarr_dirs)} test datasets")
    print(f"{'=' * 64}\n")

    all_rows = []
    row_id = 0

    for zp in zarr_dirs:
        ds = os.path.basename(zp).replace(".zarr", "")
        try:
            nodes, edges = process_dataset(zp, ds)
        except Exception as exc:
            print(f"  [{ds}] ERROR: {exc} — emitting stub")
            nodes = [(1, 0, 32, 128, 128), (2, 1, 32, 128, 128)]
            edges = [(1, 2)]

        for nid, t, z, y, x in nodes:
            all_rows.append([row_id, ds, "node", nid, t, z, y, x, -1, -1])
            row_id += 1
        for src, tgt in edges:
            all_rows.append([row_id, ds, "edge", -1, -1, -1, -1, -1, src, tgt])
            row_id += 1

    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in cols[3:]:
        df[c] = df[c].astype(int)

    df.to_csv(output_path, index=False)
    nn = (df.row_type == "node").sum()
    ne = (df.row_type == "edge").sum()
    nd = df.dataset.nunique()
    
    print(f"{'=' * 64}")
    print(f"  Submission saved → {output_path}")
    print(f"  {nd} datasets  |  {nn:,} nodes  |  {ne:,} edges  |  {len(df):,} rows")
    print(f"{'=' * 64}\n")
    return df


if __name__ == "__main__":
    TEST_DIR  = "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"
    OUT_CSV   = "submission.csv"

    t_wall = time.time()
    submission = create_submission(TEST_DIR, OUT_CSV)
    print(f"Total wall time: {(time.time() - t_wall) / 60:.1f} min")
    print(submission.head(20))
