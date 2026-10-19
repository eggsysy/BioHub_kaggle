"""Auto-generated from best_ensemble_0.935.ipynb cell 6 -- do not edit."""
import os
from pathlib import Path

COMPETITION = 'biohub-cell-tracking-during-development'
COMP_DIR_CANDIDATES = [Path(f'/kaggle/input/competitions/{COMPETITION}'), Path(f'/kaggle/input/{COMPETITION}')]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])
TEST_DIR = COMP_DIR / 'test'
WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
REPO_DIR = WORKING_DIR / 'tracking_repo'
SUBMISSION_PATH = WORKING_DIR / 'submission.csv'
RUN_STATS_PATH = WORKING_DIR / 'run_stats.csv'
METHOD = 'unet_transformer'
WEIGHTS_RELATIVE = f'weights/{METHOD}/split_0/edge_predictor_best.pth'
EXPERIMENT_TAG = 'selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap'
TARGET_ARTIFACT_SLUG = os.environ.get('BIOHUB_TARGET_ARTIFACT_SLUG', 'biohub-tracking-support-pack-50ep-v1')
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get('BIOHUB_PRIMARY_ARTIFACT_MANIFEST', '/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json'))
ALLOW_ARTIFACT_FALLBACK = os.environ.get('BIOHUB_ALLOW_ARTIFACT_FALLBACK', '0') != '0'
DET_THRESHOLD = float(os.environ.get('BIOHUB_DET_THRESHOLD', '0.99'))
UNET_BATCH_SIZE = int(os.environ.get('BIOHUB_UNET_BATCH_SIZE', '4'))
USE_ILP = os.environ.get('BIOHUB_USE_ILP', '1') != '0'
ILP_EDGE_WEIGHT = float(os.environ.get('BIOHUB_ILP_EDGE_WEIGHT', '-1.0'))
ILP_APPEARANCE_WEIGHT = float(os.environ.get('BIOHUB_ILP_APPEARANCE_WEIGHT', '0.1'))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get('BIOHUB_ILP_DISAPPEARANCE_WEIGHT', '0.1'))
ILP_DIVISION_WEIGHT = float(os.environ.get('BIOHUB_ILP_DIVISION_WEIGHT', '1.0'))
SLICE = ''
ALLOW_PIP_INSTALL = os.environ.get('BIOHUB_ALLOW_PIP_INSTALL', '0') != '0'
RUN_OUTPUT_DIAGNOSTICS = os.environ.get('BIOHUB_RUN_OUTPUT_DIAGNOSTICS', '1') != '0'
OUTPUT_EDGE_MAX_UM = float(os.environ.get('BIOHUB_OUTPUT_EDGE_MAX_UM', '14.0'))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get('BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME', '1') != '0'
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get('BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR', '1') != '0'
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get('BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR', '0') != '0'
OUTPUT_PRUNE_ISOLATED = os.environ.get('BIOHUB_OUTPUT_PRUNE_ISOLATED', '1') != '0'
OUTPUT_MOTION_RELINK = os.environ.get('BIOHUB_OUTPUT_MOTION_RELINK', '1') != '0'
MOTION_RELINK_TIGHT_UM = float(os.environ.get('BIOHUB_MOTION_RELINK_TIGHT_UM', '6.0'))
MOTION_RELINK_RELAXED_UM = float(os.environ.get('BIOHUB_MOTION_RELINK_RELAXED_UM', '10.0'))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get('BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT', '0.5'))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get('BIOHUB_MOTION_RELINK_LEARNED_BONUS', '0.75'))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get('BIOHUB_MOTION_RELINK_MAX_FRAME_NODES', '2600'))
OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get('BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER', '0') != '0'
DIV_PARENT_MAX_UM = float(os.environ.get('BIOHUB_DIV_PARENT_MAX_UM', '10.5'))
DIV_SISTER_MAX_UM = float(os.environ.get('BIOHUB_DIV_SISTER_MAX_UM', '8.0'))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get('BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD', '1') != '0'
OUTPUT_GAP_CLOSE = os.environ.get('BIOHUB_OUTPUT_GAP_CLOSE', '1') != '0'
GAP_CLOSE_MAX_GAP = int(os.environ.get('BIOHUB_GAP_CLOSE_MAX_GAP', '1'))
GAP_CLOSE_UM = float(os.environ.get('BIOHUB_GAP_CLOSE_UM', '6.0'))
GAP_DENSITY_ADAPTIVE = os.environ.get('BIOHUB_GAP_DENSITY_ADAPTIVE', '0') != '0'
GAP_DENSITY_REFERENCE_UM = float(os.environ.get('BIOHUB_GAP_DENSITY_REFERENCE_UM', '6.5'))
GAP_DENSITY_GAIN = float(os.environ.get('BIOHUB_GAP_DENSITY_GAIN', '0.040'))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get('BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM', '0.125'))
GAP_DENSITY_NEIGHBORS = int(os.environ.get('BIOHUB_GAP_DENSITY_NEIGHBORS', '3'))
GAP_CLOSE_REUSE_EXISTING = os.environ.get('BIOHUB_GAP_CLOSE_REUSE_EXISTING', '1') != '0'
GAP_CLOSE_REUSE_UM = float(os.environ.get('BIOHUB_GAP_CLOSE_REUSE_UM', '3.2'))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get('BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC', '0.05'))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get('BIOHUB_GAP_CLOSE_MAX_ADDED_ABS', '2000'))
GAP_REFINE_SYNTHETIC = os.environ.get('BIOHUB_GAP_REFINE_SYNTHETIC', '1') != '0'
GAP_REFINE_WIN_Z = int(os.environ.get('BIOHUB_GAP_REFINE_WIN_Z', '1'))
GAP_REFINE_WIN_YX = int(os.environ.get('BIOHUB_GAP_REFINE_WIN_YX', '3'))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get('BIOHUB_GAP_REFINE_MAX_SHIFT_UM', '3.2'))
OUTPUT_FILTER_SHORT_TRACKS = os.environ.get('BIOHUB_OUTPUT_FILTER_SHORT_TRACKS', '1') != '0'
OUTPUT_MIN_TRACK_LEN = int(os.environ.get('BIOHUB_OUTPUT_MIN_TRACK_LEN', '6'))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get('BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS', '1') != '0'
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get('BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE', '0') != '0'
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC', '0.10'))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN', '4'))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB', '0.82'))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM', '3.25'))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC', '0.018'))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get('BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS', '180'))
OUTPUT_LINEFIT_SMOOTH = os.environ.get('BIOHUB_OUTPUT_LINEFIT_SMOOTH', '1') != '0'
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get('BIOHUB_OUTPUT_LINEFIT_WEIGHT', '0.8'))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get('BIOHUB_OUTPUT_LINEFIT_WINDOW', '2'))
OUTPUT_GAP2_RECOVERY = os.environ.get('BIOHUB_OUTPUT_GAP2_RECOVERY', '0') != '0'
GAP2_MAX_TOTAL_UM = float(os.environ.get('BIOHUB_GAP2_MAX_TOTAL_UM', '10.2'))
GAP2_MAX_STEP_UM = float(os.environ.get('BIOHUB_GAP2_MAX_STEP_UM', '4.4'))
GAP2_MAX_LINKS_FRAC = float(os.environ.get('BIOHUB_GAP2_MAX_LINKS_FRAC', '0.0045'))
GAP2_MAX_LINKS_ABS = int(os.environ.get('BIOHUB_GAP2_MAX_LINKS_ABS', '180'))
GAP2_REQUIRE_CONTEXT = os.environ.get('BIOHUB_GAP2_REQUIRE_CONTEXT', '1') != '0'
GAP2_FRAME_FRAC_CAP = float(os.environ.get('BIOHUB_GAP2_FRAME_FRAC_CAP', '0.006'))
OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'
SAFE_DIV_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_MAX_UM', '4.7'))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_SISTER_MAX_UM', '7.2'))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get('BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM', '7.8'))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get('BIOHUB_SAFE_DIV_FRAME_FRAC_CAP', '0.008'))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get('BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP', '0.004'))
SAFE_DIV_DIVERGE_UM = float(os.environ.get('BIOHUB_SAFE_DIV_DIVERGE_UM', '2.25'))
SAFE_DIV_REQUIRE_DIVERGENCE = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE', '1') != '0'
SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN', '1') != '0'
USE_DEEPCENTER_VETO = os.environ.get('BIOHUB_USE_DEEPCENTER_VETO', '1') != '0'
REQUIRE_DEEPCENTER_VETO = os.environ.get('BIOHUB_REQUIRE_DEEPCENTER_VETO', '1') != '0'
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get('BIOHUB_DEEPCENTER_MANIFEST_DEFAULT', '/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json')
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get('BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT', '/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt')
DEEPCENTER_RELATIVE = os.environ.get('BIOHUB_DEEPCENTER_RELATIVE', 'weights/full_frame_center/best.pt')
DEEPCENTER_GAP_VETO = os.environ.get('BIOHUB_DEEPCENTER_GAP_VETO', '1') != '0'
DEEPCENTER_SAFE_DIV_VETO = os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_VETO', '1') != '0'
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get('BIOHUB_DEEPCENTER_GAP_THRESHOLD', '0.10'))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get('BIOHUB_DEEPCENTER_EXPECTED_EPOCH', '0'))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get('BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM', '0'))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD', '0.12'))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_WIN_Z', '1'))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_WIN_YX', '2'))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get('BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES', '8'))
CONFIG_DISPLAY = {'experiment_tag': EXPERIMENT_TAG, 'method': METHOD, 'weights': WEIGHTS_RELATIVE, 'target_artifact_slug': TARGET_ARTIFACT_SLUG, 'primary_artifact_manifest': str(PRIMARY_ARTIFACT_MANIFEST), 'allow_artifact_fallback': ALLOW_ARTIFACT_FALLBACK, 'det_threshold': DET_THRESHOLD, 'unet_batch_size': UNET_BATCH_SIZE, 'use_ilp': USE_ILP, 'ilp_edge_weight': ILP_EDGE_WEIGHT, 'ilp_appearance_weight': ILP_APPEARANCE_WEIGHT, 'ilp_disappearance_weight': ILP_DISAPPEARANCE_WEIGHT, 'ilp_division_weight': ILP_DIVISION_WEIGHT, 'slice': SLICE, 'allow_pip_install': ALLOW_PIP_INSTALL, 'output_edge_max_um': OUTPUT_EDGE_MAX_UM, 'output_enforce_next_frame': OUTPUT_ENFORCE_NEXT_FRAME, 'output_single_parent_repair': OUTPUT_SINGLE_PARENT_REPAIR, 'output_single_child_repair': OUTPUT_SINGLE_CHILD_REPAIR, 'output_prune_isolated': OUTPUT_PRUNE_ISOLATED, 'output_motion_relink': OUTPUT_MOTION_RELINK, 'motion_relink_tight_um': MOTION_RELINK_TIGHT_UM, 'motion_relink_relaxed_um': MOTION_RELINK_RELAXED_UM, 'motion_relink_velocity_weight': MOTION_RELINK_VELOCITY_WEIGHT, 'motion_relink_learned_bonus': MOTION_RELINK_LEARNED_BONUS, 'motion_relink_max_frame_nodes': MOTION_RELINK_MAX_FRAME_NODES, 'output_division_geometry_filter': OUTPUT_DIVISION_GEOMETRY_FILTER, 'div_parent_max_um': DIV_PARENT_MAX_UM, 'div_sister_max_um': DIV_SISTER_MAX_UM, 'div_drop_to_single_if_bad': DIV_DROP_TO_SINGLE_IF_BAD, 'output_gap_close': OUTPUT_GAP_CLOSE, 'gap_close_max_gap': GAP_CLOSE_MAX_GAP, 'gap_close_effective_max_gap': min(GAP_CLOSE_MAX_GAP, 1), 'gap_close_um': GAP_CLOSE_UM, 'gap_density_adaptive': GAP_DENSITY_ADAPTIVE, 'gap_density_reference_um': GAP_DENSITY_REFERENCE_UM, 'gap_density_gain': GAP_DENSITY_GAIN, 'gap_density_max_step_delta_um': GAP_DENSITY_MAX_STEP_DELTA_UM, 'gap_density_neighbors': GAP_DENSITY_NEIGHBORS, 'gap_close_reuse_existing': GAP_CLOSE_REUSE_EXISTING, 'gap_close_reuse_um': GAP_CLOSE_REUSE_UM, 'gap_close_max_added_frac': GAP_CLOSE_MAX_ADDED_FRAC, 'gap_close_max_added_abs': GAP_CLOSE_MAX_ADDED_ABS, 'gap_refine_synthetic': GAP_REFINE_SYNTHETIC, 'gap_refine_win_z': GAP_REFINE_WIN_Z, 'gap_refine_win_yx': GAP_REFINE_WIN_YX, 'gap_refine_max_shift_um': GAP_REFINE_MAX_SHIFT_UM, 'output_filter_short_tracks': OUTPUT_FILTER_SHORT_TRACKS, 'output_min_track_len': OUTPUT_MIN_TRACK_LEN, 'output_keep_division_components': OUTPUT_KEEP_DIVISION_COMPONENTS, 'adaptive_short_track_rescue': ADAPTIVE_SHORT_TRACK_RESCUE, 'short_track_rescue_trigger_removed_frac': SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC, 'short_track_rescue_min_len': SHORT_TRACK_RESCUE_MIN_LEN, 'short_track_rescue_min_mean_edge_prob': SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB, 'short_track_rescue_max_mean_edge_dist_um': SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM, 'short_track_rescue_max_nodes_frac': SHORT_TRACK_RESCUE_MAX_NODES_FRAC, 'short_track_rescue_max_nodes_abs': SHORT_TRACK_RESCUE_MAX_NODES_ABS, 'output_linefit_smooth': OUTPUT_LINEFIT_SMOOTH, 'output_linefit_weight': OUTPUT_LINEFIT_WEIGHT, 'output_linefit_window': OUTPUT_LINEFIT_WINDOW, 'output_gap2_recovery': OUTPUT_GAP2_RECOVERY, 'gap2_max_total_um': GAP2_MAX_TOTAL_UM, 'gap2_max_step_um': GAP2_MAX_STEP_UM, 'gap2_max_links_frac': GAP2_MAX_LINKS_FRAC, 'gap2_max_links_abs': GAP2_MAX_LINKS_ABS, 'gap2_require_context': GAP2_REQUIRE_CONTEXT, 'gap2_frame_frac_cap': GAP2_FRAME_FRAC_CAP, 'output_safe_divisions': OUTPUT_SAFE_DIVISIONS, 'safe_div_max_um': SAFE_DIV_MAX_UM, 'safe_div_sister_max_um': SAFE_DIV_SISTER_MAX_UM, 'safe_div_existing_child_max_um': SAFE_DIV_EXISTING_CHILD_MAX_UM, 'safe_div_frame_frac_cap': SAFE_DIV_FRAME_FRAC_CAP, 'safe_div_global_frac_cap': SAFE_DIV_GLOBAL_FRAC_CAP, 'use_deepcenter_add_only_gate': USE_DEEPCENTER_VETO, 'deepcenter_gap_add_gate': DEEPCENTER_GAP_VETO, 'deepcenter_safe_div_add_gate': DEEPCENTER_SAFE_DIV_VETO, 'deepcenter_gap_threshold': DEEPCENTER_GAP_THRESHOLD, 'deepcenter_expected_epoch': DEEPCENTER_EXPECTED_EPOCH, 'deepcenter_gap_confirm_min_span_um': DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM, 'deepcenter_safe_div_threshold': DEEPCENTER_SAFE_DIV_THRESHOLD, 'deepcenter_checkpoint_default': DEEPCENTER_CHECKPOINT_DEFAULT}
