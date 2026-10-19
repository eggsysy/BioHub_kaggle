"""Measure the geometry of every annotated division in the training set.

The safe-division rule gates candidates on parent->daughter distance, sister
separation and divergence. Those thresholds were tuned against 6 divisions.
Here we measure all 151 directly, in micrometres, to see whether the shipped
bounds are anywhere near the truth.
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

GT_DIR = Path(__file__).parent / "gt"
SCALE = np.array([1.625, 0.40625, 0.40625])


def load(path: Path):
    g = zarr.open(str(path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    pos = np.stack([np.asarray(g[f"nodes/props/{k}/values"]) for k in "zyx"], axis=1)
    t = np.asarray(g["nodes/props/t/values"])
    edges = np.asarray(g["edges/ids"])
    idx = {int(v): i for i, v in enumerate(ids)}
    return ids, t, pos, edges, idx


parent_daughter, sister, divergence, daughter_dt = [], [], [], []
n_div = 0
frac_of_movie = []

for path in sorted(GT_DIR.glob("*.geff")):
    ids, t, pos, edges, idx = load(path)
    if len(edges) == 0:
        continue
    children = defaultdict(list)
    for s, d in edges:
        children[int(s)].append(int(d))

    n_frames = int(t.max()) + 1
    for src, kids in children.items():
        if len(kids) < 2:
            continue
        n_div += 1
        si = idx[src]
        p = pos[si] * SCALE
        ks = kids[:2]
        cpos = [pos[idx[k]] * SCALE for k in ks]
        parent_daughter.extend(float(np.linalg.norm(p - c)) for c in cpos)
        sister.append(float(np.linalg.norm(cpos[0] - cpos[1])))
        daughter_dt.extend(int(t[idx[k]]) - int(t[si]) for k in ks)
        frac_of_movie.append(int(t[si]) / max(n_frames - 1, 1))

        # divergence: how much further apart the grandchildren are
        gk = [children.get(k, []) for k in ks]
        if all(len(g) >= 1 for g in gk):
            g0 = pos[idx[gk[0][0]]] * SCALE
            g1 = pos[idx[gk[1][0]]] * SCALE
            divergence.append(float(np.linalg.norm(g0 - g1)) - sister[-1])


def show(name, arr, shipped=None):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        print(f"{name}: no data")
        return
    qs = np.percentile(a, [5, 25, 50, 75, 90, 95, 99])
    print(f"{name:22s} n={a.size:4d}  min={a.min():6.2f}  "
          f"p5={qs[0]:5.2f} p25={qs[1]:5.2f} med={qs[2]:5.2f} "
          f"p75={qs[3]:5.2f} p90={qs[4]:5.2f} p95={qs[5]:5.2f} p99={qs[6]:5.2f}  max={a.max():6.2f}")
    if shipped is not None:
        keep = float((a <= shipped).mean())
        print(f"{'':22s}  shipped threshold {shipped} um -> admits {keep:.1%} of real divisions")


print(f"annotated divisions found: {n_div}\n")
show("parent->daughter um", parent_daughter, shipped=7.0)     # SAFE_DIV_MAX_UM
show("sister separation um", sister, shipped=12.0)            # SAFE_DIV_SISTER_MAX_UM
show("grandchild divergence", divergence, shipped=None)       # SAFE_DIV_DIVERGE_UM
print()
print("daughter frame offset:", Counter(daughter_dt).most_common())
fm = np.asarray(frac_of_movie)
print(f"division timing (fraction through movie): med={np.median(fm):.2f} "
      f"p10={np.percentile(fm,10):.2f} p90={np.percentile(fm,90):.2f}")


# --- joint pass rate of the shipped gates, applied together ------------------
print("\n" + "=" * 74)
print("Shipped safe-division gates applied JOINTLY to the 151 real divisions")
print("=" * 74)

pd_arr = np.asarray(parent_daughter, dtype=float).reshape(-1, 2)  # two daughters per division
sis = np.asarray(sister, dtype=float)
div = np.asarray(divergence, dtype=float)

gates = {
    "both daughters <= SAFE_DIV_MAX_UM (7.0)": (pd_arr <= 7.0).all(axis=1),
    "sister <= SAFE_DIV_SISTER_MAX_UM (12.0)": sis <= 12.0,
}
for name, mask in gates.items():
    print(f"  {name:45s} passes {mask.mean():6.1%}")

both = gates["both daughters <= SAFE_DIV_MAX_UM (7.0)"] & gates["sister <= SAFE_DIV_SISTER_MAX_UM (12.0)"]
print(f"  {'both geometric gates together':45s} passes {both.mean():6.1%}")

# divergence is only measurable where grandchildren exist
dmask = np.zeros(len(sis), dtype=bool)
dmask[: len(div)] = div >= 2.25
print(f"  {'+ REQUIRE_DIVERGENCE (>= 2.25 um)':45s} passes "
      f"{(both[: len(div)] & (div >= 2.25)).mean():6.1%}  <-- ceiling on division recall")

print("\nBounds that would admit 95% of real divisions:")
print(f"  SAFE_DIV_MAX_UM        >= {np.percentile(pd_arr.max(axis=1), 95):.1f}")
print(f"  SAFE_DIV_SISTER_MAX_UM >= {np.percentile(sis, 95):.1f}")
print(f"  SAFE_DIV_DIVERGE_UM    <= {np.percentile(div, 5):.2f}")
