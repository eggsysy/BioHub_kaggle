import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = nb['cells'][:8] # Keep 0-7

# Cell 9 (Index 9 in original)
cell9 = nb['cells'][9]
source9 = "".join(cell9['source'])
lines9 = source9.split('\n')
keep9 = []
for line in lines9:
    if "geffs = sorted(" in line:
        break
    keep9.append(line)
cell9['source'] = [line + '\n' for line in keep9]
new_cells.append(cell9)

# Cell 12 (Index 12 in original)
cell12 = nb['cells'][12]
source12 = "".join(cell12['source'])
lines12 = source12.split('\n')
keep12 = []
for line in lines12:
    if "if VALIDATOR_ENABLE and val_stems:" in line:
        break
    keep12.append(line)
cell12['source'] = [line + '\n' for line in keep12]
new_cells.append(cell12)

# Optuna Cell
optuna_source = """import optuna
import os
from pathlib import Path

print('Starting Optuna Optimization...')

TRAIN_DIR = COMP_DIR / 'train'
val_stems = ['44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132']
VALIDATOR_ENABLE = True
VALIDATOR_MATCH_RADIUS_UM = 7.0

def objective(trial):
    bidir_weight = trial.suggest_float('bidir_weight', 0.0, 1.0)
    safe_div_max = trial.suggest_float('safe_div_max', 3.0, 15.0)
    sister_max = trial.suggest_float('sister_max', 3.0, 15.0)
    
    os.environ['BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT'] = str(bidir_weight)
    os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = str(safe_div_max)
    os.environ['BIOHUB_SAFE_DIV_SISTER_MAX_UM'] = str(sister_max)
    
    rows_this_config = []
    for stem in val_stems:
        gt_path = TRAIN_DIR / f'{stem}.geff'
        pred_path = Path(f'/kaggle/input/biohub-948-sew20/tracking_repo/predictions/unknown/unet_transformer_val/split_0/{stem}.geff')
        if not gt_path.exists() or not pred_path.exists():
            continue
        
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)
        
        pred_graph = graph_from_geff(pred_path)
        raw_nodes_by_id = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row['node_id'])
            raw_nodes_by_id[node_id] = {'node_id': node_id, 't': int(row['t']), 'z': float(row['z']), 'y': float(row['y']), 'x': float(row['x'])}
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get('edge_prob') if hasattr(row, 'get') else None
            raw_edges.append({'source_id': int(row['source_id']), 'target_id': int(row['target_id']), 'edge_prob': None if edge_prob is None else float(edge_prob)})
            
        _real_test_dir = TEST_DIR
        globals()['TEST_DIR'] = TRAIN_DIR
        try:
            processed_nodes, processed_edges, _ = filter_output_graph(
                raw_nodes_by_id, raw_edges, dataset=stem, deepcenter_bundle=globals().get('DEEPCENTER_VETO_DETECTOR')
            )
        except Exception as e:
            print('Error in filter:', e)
            return 0.0
        finally:
            globals()['TEST_DIR'] = _real_test_dir
            
        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
        pred_edges_plain = [(int(e['source_id']), int(e['target_id'])) for e in processed_edges]
        
        row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
        rows_this_config.append(row)
        
    if rows_this_config:
        summary = aggregate_official(rows_this_config)
        return summary['proxy_score']
    return 0.0

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('BEST PARAMETERS:', study.best_params)
print('BEST SCORE:', study.best_value)

import pandas as pd
df = study.trials_dataframe()
df.to_csv('optuna_results.csv', index=False)
"""
optuna_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in optuna_source.split('\n')]
}
new_cells.append(optuna_cell)
nb['cells'] = new_cells

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'w') as f:
    json.dump(nb, f)
