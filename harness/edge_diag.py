"""Where does the edge score actually go: missed links (FN) or wrong links (FP)?

The metric ignores predicted edges that touch no annotated cell, so over-linking
in unannotated regions is free while under-linking always costs an FN. If FN
dominates, the pipeline should link MORE aggressively. This ratio determines
which direction every future change should push.
"""
import sys
sys.path.insert(0, "harness"); sys.path.insert(0, "official_repo/src")
from pathlib import Path
import replay

CACHE = Path("cache/pred_cache")
stems = replay.discover_stems(CACHE)
r = replay.score_stems_parallel(stems, {}, CACHE, workers=8)

tp = sum(x["edge_tp"] for x in r["per_sample"])
fp = sum(x["edge_fp"] for x in r["per_sample"])
fn = sum(x["edge_fn"] for x in r["per_sample"])
print(f"\nEDGES over {len(stems)} samples")
print(f"  TP = {tp:,}")
print(f"  FP = {fp:,}   ({fp/(fp+fn):.1%} of errors)")
print(f"  FN = {fn:,}   ({fn/(fp+fn):.1%} of errors)")
print(f"  jaccard = {tp/(tp+fp+fn):.4f}")
print(f"  precision = {tp/(tp+fp):.4f}   recall = {tp/(tp+fn):.4f}")
print(f"\nIf all FP vanished: J = {tp/(tp+fn):.4f}  (+{tp/(tp+fn)-tp/(tp+fp+fn):.4f})")
print(f"If all FN vanished: J = {tp/(tp+fp):.4f}  (+{tp/(tp+fp)-tp/(tp+fp+fn):.4f})")

print(f"\nDIVISIONS: tp={r['division_tp']} fp={r['division_fp']} fn={r['division_fn']}"
      f"  precision={r['division_tp']/max(r['division_tp']+r['division_fp'],1):.3f}"
      f"  recall={r['division_tp']/max(r['division_tp']+r['division_fn'],1):.3f}")

# node-count headroom
import csv, json
inv = {x["stem"]: float(x["estimated_number_of_nodes"])
       for x in csv.DictReader(open("gt/gt_inventory.csv"))}
tot_pred = sum(x["num_pred_nodes"] for x in r["per_sample"])
tot_true = sum(inv[x["stem"]] for x in r["per_sample"])
print(f"\nNODES: predicted={tot_pred:,}  estimated true={tot_true:,.0f}  "
      f"ratio={tot_pred/tot_true:.3f}")
print(f"  current node-count multiplier ~= {1 - 0.1*(tot_pred-tot_true)/tot_true:.4f}")
over = [(x['stem'], x['num_pred_nodes']/inv[x['stem']]) for x in r["per_sample"]]
over.sort(key=lambda t: -t[1])
print("  most over-predicted:", [(s, f"{v:.2f}") for s, v in over[:5]])
print("  most under-predicted:", [(s, f"{v:.2f}") for s, v in over[-5:]])
