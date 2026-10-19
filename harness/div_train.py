"""Train the division classifier, validated on a genuinely held-out embryo.

52 positives among 7.36M candidates, so this is a ranking problem, not a
classification one: we only ever add the top-K forks. Trained on one embryo
and scored on the other -- the discipline the post-processing tuning lacked,
where I split the same 50 samples I had tuned on and duly failed to detect
overfitting.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATS = ["d_parent_child", "d_parent_cand", "d_sister", "diverge",
         "prob_PC1", "q_parent_dist", "q_parent_prob",  # prob_PQ is NaN by construction
         "q_parent_nkids", "q_has_parent", "dens_P", "dens_C1", "dens_Q",
         "dc_P", "dc_C1", "dc_Q", "n_kids_C1", "n_kids_Q",
         "t_frac", "frame_nodes"]

df = pd.read_csv("div_candidates.csv")
df["embryo"] = df["stem"].str.split("_").str[0]
print(f"{len(df):,} candidates | positives: {df.y.sum()} "
      f"| by embryo: {df.groupby('embryo').y.sum().to_dict()}")

for train_emb, test_emb in (("6bba", "44b6"), ("44b6", "6bba")):
    tr, te = df[df.embryo == train_emb], df[df.embryo == test_emb]
    if tr.y.sum() < 5 or te.y.sum() < 5:
        print(f"\n{train_emb} -> {test_emb}: too few positives, skipping")
        continue
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, class_weight={0: 1.0, 1: 500.0},
        early_stopping=False, random_state=0)
    clf.fit(tr[FEATS], tr.y)
    s = clf.predict_proba(te[FEATS])[:, 1]

    print(f"\n=== train {train_emb} ({tr.y.sum()} pos) -> test {test_emb} "
          f"({te.y.sum()} pos, {len(te):,} candidates)")
    order = np.argsort(-s)
    yt = te.y.values[order]
    n_samples = te.stem.nunique()
    print(f"{'top-K':>8s} {'TP':>4s} {'precision':>10s} {'recall':>8s}   (K per sample)")
    for K in (25, 50, 75, 100, 150, 250, 400):
        tp = int(yt[:K].sum())
        print(f"{K:8d} {tp:4d} {tp/K:10.3f} {tp/te.y.sum():8.3f}   {K/n_samples:.1f}")

    # what the shipped hand-tuned rule achieves, for reference
    print("  (shipped rule on all 50 samples: 14 TP / 75 forks = 0.187 precision,"
          " 0.192 recall)")

    imp = sorted(zip(FEATS, clf.feature_importances_ if hasattr(clf, "feature_importances_")
                     else [0]*len(FEATS)), key=lambda t: -t[1])[:8] \
        if hasattr(clf, "feature_importances_") else []
    if imp:
        print("  top features:", ", ".join(f"{k}={v:.3f}" for k, v in imp))
