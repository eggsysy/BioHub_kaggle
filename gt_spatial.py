"""Is the ground-truth annotation spatially biased, or a random subset?

The oracle prune showed a huge ceiling, but it needs to know WHICH predicted
nodes match GT. If annotation is a uniform random sample of real cells, that
is unknowable and the ceiling is unreachable. If annotators instead labelled a
sub-region or a particular depth range, that bias is learnable from the 199
training samples and part of the ceiling becomes accessible.
"""
import numpy as np, zarr
from pathlib import Path

GT = Path("gt")
zs, ys, xs, ts, spread = [], [], [], [], []
for p in sorted(GT.glob("*.geff")):
    g = zarr.open(str(p), mode="r")
    z = np.asarray(g["nodes/props/z/values"]).astype(float)
    y = np.asarray(g["nodes/props/y/values"]).astype(float)
    x = np.asarray(g["nodes/props/x/values"]).astype(float)
    t = np.asarray(g["nodes/props/t/values"]).astype(float)
    if len(z) == 0:
        continue
    # volume is (T,64,256,256); normalise each axis to [0,1]
    zs.append(z / 63.0); ys.append(y / 255.0); xs.append(x / 255.0)
    ts.append(t / max(t.max(), 1))
    spread.append((np.ptp(z) / 63.0, np.ptp(y) / 255.0, np.ptp(x) / 255.0))

for name, arr in (("z", np.concatenate(zs)), ("y", np.concatenate(ys)),
                  ("x", np.concatenate(xs)), ("t", np.concatenate(ts))):
    q = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"GT {name} (normalised 0-1): p1={q[0]:.2f} p5={q[1]:.2f} p25={q[2]:.2f} "
          f"med={q[3]:.2f} p75={q[4]:.2f} p95={q[5]:.2f} p99={q[6]:.2f}")

sp = np.array(spread)
print(f"\nper-sample GT bounding-box extent (fraction of full axis):")
print(f"  z: med={np.median(sp[:,0]):.2f}  y: med={np.median(sp[:,1]):.2f}  x: med={np.median(sp[:,2]):.2f}")
frac = np.median(sp[:,0] * sp[:,1] * sp[:,2])
print(f"  median GT bbox covers {frac:.1%} of the volume")
print("\nIf GT filled the volume uniformly these would all be ~1.0 and the")
print("percentiles would be flat -- annotation would be unlearnable.")
