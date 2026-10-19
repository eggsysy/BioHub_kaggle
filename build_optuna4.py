import json

with open('/Users/aryanbadmera/Desktop/BioHub/sew20_kernel/biohub-948-sew20.ipynb', 'r') as f:
    nb = json.load(f)

# Let's find the required cells
imports_cell = nb['cells'][1]  # Usually pip installs or imports
config_cell = nb['cells'][2]   # Usually configuration
imports2_cell = nb['cells'][3] # More imports
# Actually, let's just keep the cells up to cell 5, which are mostly imports, setup, and config.
# Cell 6 is Inference.
# We will identify cells by looking at their content.

new_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Inference Phase" in source or "datasets = " in source:
            print("Skipping Inference Phase")
            continue
        if "prediction completed in" in source or "for shard_index in range(worker_count):" in source:
            print("Skipping Predictor loop")
            continue
        if "LOCAL VALIDATOR" in source and "filter_output_graph" in source:
            print("Skipping original Validator")
            continue
        if "_guard_submission =" in source or "retention_guard" in source:
            print("Skipping Guard")
            continue
        if "submission_rows" in source or "pd.DataFrame(" in source and "row_type" in source:
            print("Skipping final submission")
            continue
    new_cells.append(cell)

optuna_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import optuna\n",
        "import os\n",
        "from pathlib import Path\n",
        "print('Starting Optuna Optimization...')\n",
        "\n",
        "test_stems = []\n",
        "def objective(trial):\n",
        "    bidir_weight = trial.suggest_float('bidir_weight', 0.0, 1.0)\n",
        "    safe_div_max = trial.suggest_float('safe_div_max', 3.0, 15.0)\n",
        "    sister_max = trial.suggest_float('sister_max', 3.0, 15.0)\n",
        "    \n",
        "    os.environ['BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT'] = str(bidir_weight)\n",
        "    os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = str(safe_div_max)\n",
        "    os.environ['BIOHUB_SAFE_DIV_SISTER_MAX_UM'] = str(sister_max)\n",
        "    \n",
        "    rows_this_config = []\n",
        "    # We iterate over val_stems which is defined in Cell 8\n",
        "    for stem in val_stems:\n",
        "        gt_path = TRAIN_DIR / f'{stem}.geff'\n",
        "        # Read pred from the attached kernel's output!\n",
        "        pred_path = Path(f'/kaggle/input/biohub-948-sew20/tracking_repo/predictions/unknown/unet_transformer_val/split_0/{stem}.geff')\n",
        "        if not gt_path.exists() or not pred_path.exists():\n",
        "            continue\n",
        "        \n",
        "        gt_graph = graph_from_geff(gt_path)\n",
        "        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)\n",
        "        t_true = read_estimated_true_node_count(gt_path)\n",
        "        \n",
        "        pred_graph = graph_from_geff(pred_path)\n",
        "        raw_nodes_by_id = {}\n",
        "        for row in pred_graph.node_attrs().iter_rows(named=True):\n",
        "            node_id = int(row['node_id'])\n",
        "            raw_nodes_by_id[node_id] = {'node_id': node_id, 't': int(row['t']), 'z': float(row['z']), 'y': float(row['y']), 'x': float(row['x'])}\n",
        "        raw_edges = []\n",
        "        for row in pred_graph.edge_attrs().iter_rows(named=True):\n",
        "            raw_edges.append({'source_id': int(row['source_id']), 'target_id': int(row['target_id']), 'edge_prob': float(row.get('edge_prob', 0))})\n",
        "            \n",
        "        _real_test_dir = TEST_DIR\n",
        "        globals()['TEST_DIR'] = TRAIN_DIR\n",
        "        try:\n",
        "            processed_nodes, processed_edges, _ = filter_output_graph(raw_nodes_by_id, raw_edges, dataset=stem, deepcenter_bundle=globals().get('DEEPCENTER_VETO_DETECTOR'))\n",
        "        except Exception as e:\n",
        "            return 0.0\n",
        "        finally:\n",
        "            globals()['TEST_DIR'] = _real_test_dir\n",
        "            \n",
        "        pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)\n",
        "        pred_edges_plain = [(int(e['source_id']), int(e['target_id'])) for e in processed_edges]\n",
        "        \n",
        "        row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)\n",
        "        rows_this_config.append(row)\n",
        "        \n",
        "    if rows_this_config:\n",
        "        summary = aggregate_official(rows_this_config)\n",
        "        return summary['proxy_score']\n",
        "    return 0.0\n",
        "\n",
        "study = optuna.create_study(direction='maximize')\n",
        "study.optimize(objective, n_trials=50)\n",
        "print('BEST PARAMETERS:', study.best_params)\n",
        "print('BEST SCORE:', study.best_value)\n",
        "import pandas as pd\n",
        "df = study.trials_dataframe()\n",
        "df.to_csv('optuna_results.csv', index=False)\n"
    ]
}

new_cells.append(optuna_cell)
nb['cells'] = new_cells

with open('/Users/aryanbadmera/Desktop/BioHub/optuna_kernel/optuna-tuner.ipynb', 'w') as f:
    json.dump(nb, f)

