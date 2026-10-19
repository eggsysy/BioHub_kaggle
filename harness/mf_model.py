"""Pairwise vs multi-frame association -- the controlled comparison.

Same features, same capacity, same training budget. The ONLY difference is how
many frames the model may attend over when scoring a link between t and t+1:

    W=2   sees only frames t, t+1        (what the shipped model does)
    W=4   also sees t-1 and t+2          (temporal context)

If W>2 wins on a held-out embryo, the pairwise ceiling is real and the GCP
credits have a justified target. If not, the hypothesis dies cheaply.
"""
from __future__ import annotations
import math, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
MAX_LINK_UM = 12.0          # candidate radius; real links are well inside this


class AssocNet(nn.Module):
    """Transformer over all nodes in a temporal window; scores t->t+1 pairs."""

    def __init__(self, d=96, nhead=4, layers=3):
        super().__init__()
        self.inp = nn.Linear(7, d)          # z,y,x (um, centred), dt, and 3 local-density feats
        enc = nn.TransformerEncoderLayer(d, nhead, d * 4, batch_first=True,
                                         dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pair = nn.Sequential(
            nn.Linear(2 * d + 4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, feats, mask, src_idx, tgt_idx, rel):
        h = self.enc(self.inp(feats), src_key_padding_mask=~mask)
        a = h[0, src_idx]
        b = h[0, tgt_idx]
        return self.pair(torch.cat([a, b, rel], dim=-1)).squeeze(-1)


def frame_feats(fr, t_ref):
    """Position in um (centred) + dt + local density at 3 scales."""
    p = fr["pos"] * SCALE
    c = p.mean(0, keepdims=True)
    q = p - c
    d = np.linalg.norm(q, axis=1, keepdims=True)
    dens = []
    from scipy.spatial import cKDTree
    tree = cKDTree(p)
    for k in (2, 5, 10):
        dd, _ = tree.query(p, k=min(k + 1, len(p)))
        dd = np.atleast_2d(dd)
        dens.append(dd[:, 1:].mean(1, keepdims=True) if dd.shape[1] > 1
                    else np.full((len(p), 1), 20.0))
    dt = np.full((len(p), 1), fr["t"] - t_ref, dtype=np.float32)
    return np.concatenate([q / 50.0, dt, *[x / 20.0 for x in dens]], axis=1).astype(np.float32)


def make_examples(frames, gt_edges, W, max_pairs=4000):
    """One example per (t, t+1) transition, with a window of W frames."""
    ts = sorted(frames)
    out = []
    from scipy.spatial import cKDTree
    for i in range(len(ts) - 1):
        t, t1 = ts[i], ts[i + 1]
        if t1 != t + 1:
            continue
        half = (W - 2) // 2
        win = [x for x in range(t - half, t + 2 + (W - 2 - half)) if x in frames]
        if t not in win or t1 not in win:
            continue
        fs = [dict(frames[x], t=x) for x in win]
        feats = np.concatenate([frame_feats(f, t) for f in fs], 0)
        offs, o = {}, 0
        for f, x in zip(fs, win):
            offs[x] = o; o += len(f["ids"])
        A, B = frames[t], frames[t1]
        pa = A["pos"] * SCALE; pb = B["pos"] * SCALE
        tree = cKDTree(pb)
        si, ti, lab, rel = [], [], [], []
        for a in range(len(A["ids"])):
            for b in tree.query_ball_point(pa[a], MAX_LINK_UM):
                ga, gb = A["gt"][a], B["gt"][b]
                if ga < 0 and gb < 0:
                    continue                      # unannotated on both sides: masked out
                si.append(offs[t] + a); ti.append(offs[t1] + b)
                lab.append(1.0 if (ga >= 0 and gb >= 0 and (ga, gb) in gt_edges) else 0.0)
                d = pb[b] - pa[a]
                rel.append([*(d / 10.0), np.linalg.norm(d) / 10.0])
        if not si or sum(lab) == 0:
            continue
        if len(si) > max_pairs:
            keep = np.random.choice(len(si), max_pairs, replace=False)
            si = [si[k] for k in keep]; ti = [ti[k] for k in keep]
            lab = [lab[k] for k in keep]; rel = [rel[k] for k in keep]
        out.append((feats, np.array(si), np.array(ti),
                    np.array(lab, np.float32), np.array(rel, np.float32)))
    return out


def run(W, train_ex, val_ex, dev, steps=1500, seed=0):
    torch.manual_seed(seed)
    m = AssocNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    m.train()
    for s in range(steps):
        f, si, ti, y, rel = train_ex[np.random.randint(len(train_ex))]
        feats = torch.from_numpy(f)[None].to(dev)
        mask = torch.ones(1, f.shape[0], dtype=torch.bool, device=dev)
        logit = m(feats, mask, torch.from_numpy(si).to(dev),
                  torch.from_numpy(ti).to(dev), torch.from_numpy(rel).to(dev))
        yt = torch.from_numpy(y).to(dev)
        pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=dev).clamp(1, 50)
        loss = F.binary_cross_entropy_with_logits(logit, yt, pos_weight=pw)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()

    m.eval()
    from sklearn.metrics import average_precision_score
    P, Y = [], []
    with torch.no_grad():
        for f, si, ti, y, rel in val_ex:
            feats = torch.from_numpy(f)[None].to(dev)
            mask = torch.ones(1, f.shape[0], dtype=torch.bool, device=dev)
            lg = m(feats, mask, torch.from_numpy(si).to(dev),
                   torch.from_numpy(ti).to(dev), torch.from_numpy(rel).to(dev))
            P.append(torch.sigmoid(lg).cpu().numpy()); Y.append(y)
    P, Y = np.concatenate(P), np.concatenate(Y)
    return average_precision_score(Y, P), float((Y == 1).mean())


if __name__ == "__main__":
    data = pickle.load(open("mf_data.pkl", "rb"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr_stems = [s for s in data if s.startswith("6bba")]
    va_stems = [s for s in data if s.startswith("44b6")]
    print(f"train {len(tr_stems)} 6bba | validate {len(va_stems)} 44b6 (held out) | device {dev}")

    for W in (2, 4):
        t0 = time.time()
        tr = [e for s in tr_stems for e in make_examples(*data[s], W)]
        va = [e for s in va_stems for e in make_examples(*data[s], W)]
        ap, base = run(W, tr, va, dev)
        print(f"W={W}  window={W} frames  train_ex={len(tr):5d} val_ex={len(va):5d}  "
              f"val_AP={ap:.4f}  (positive rate {base:.4f})  [{time.time()-t0:.0f}s]",
              flush=True)
