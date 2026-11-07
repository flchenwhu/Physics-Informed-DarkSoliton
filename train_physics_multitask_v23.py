# !/usr/bin/env python3
# -*- coding: utf-8 -*-
###########flchen 2026.11 ###############
"""
train_physics_multitask.py
==========================
"""

from pathlib import Path
import copy
import ast
import json
import logging
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dark_soliton_dataloader import (
    DarkSolitonDataset,
    make_dataloader,
    compute_dataset_mean_std,
)

# 复用 Stage 2 的 backbone / localization head / protocol
import train_detector_baseline as detector_base



# =============================================================================
# 1. Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val_v2.csv"
TEST_CSV = DATA_DIR / "test.csv"
NORMALIZATION_SOURCE_CSV = TRAIN_CSV

# MODEL_NAME = "simple_cnn"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
MODEL_NAME = "fasternet_t0"

NUM_CLASSES = 3
CLASS_NAMES = ["no_excitation", "single_excitation", "other_excitations"]

SEED = 2027    # 跑3轮随机种子   取mean+std  (42  90  2027)
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 70
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

USE_CLASS_WEIGHTS = True

# Multi-task loss.
# Original V1 behavior is preserved when the new switches below are OFF.
LAMBDA_CLASSIFICATION = 1.0
LAMBDA_LOCALIZATION = 1.0
LAMBDA_HEATMAP = 1.0
LAMBDA_OFFSET = 1.0

# =========================================================================
# Stage-4 V2 multi-task optimization switches
# =========================================================================
# Recommended staged use for the controlled Stage-4 V2 line:
#   Phase B1 (default in this file):
#       - restore the original Stage-2/3 CenterNet heatmap loss reduction;
#       - bound the learnable physics-fusion strength alpha;
#       - diagnose task gradients separately on backbone / physics / shared;
#       - make LR scheduling follow the same validation JointScore as
#         checkpoint selection and early stopping.
#       - keep PCGrad / adapters / warm start / differential LR / EMA OFF.
#
#   Phase B2: only if modular diagnostics show persistent, meaningful
#             gradient conflicts, enable PCGrad as a controlled ablation.

# A. Heatmap loss reduction.
# Controlled default: False -> restore the ORIGINAL Stage-2/3
# detector_base.ModifiedFocalLoss reduction for strict comparability.
# The V2-A per-image alternative is retained only as an ablation switch.
USE_PER_IMAGE_HEATMAP_NORMALIZATION = False

# B. Gradient conflict diagnostics.
USE_GRADIENT_DIAGNOSTICS = True
GRAD_DIAGNOSTIC_INTERVAL = 10  # every N train batches; 1 = every batch
# Diagnose all scopes on the SAME sampled batch:
#   backbone    : shared 2D visual backbone only
#   physics_cls : V2.2 classification physics encoder + global late fusion
#   physics_loc : localization physics encoder + x-aligned fusion
#   physics     : union of both physics paths
#   shared      : backbone + all physics modules
GRAD_DIAGNOSTIC_SCOPES = ("backbone", "physics_cls", "physics_loc", "physics", "shared")
# Legacy single-scope name retained for old checkpoint/log compatibility.
GRAD_DIAGNOSTIC_SCOPE = "shared"
GRAD_DIAGNOSTIC_EPS = 1e-12

# C. PCGrad switch. The implementation treats classification as task-1 and
#    localization(+optional physics auxiliary loss) as task-2. Head-only
#    parameters keep their own task gradients; projection acts only where both
#    task gradients exist. For numerical clarity AMP is automatically disabled
#    when PCGrad is ON. Under PHYSICS_TASK_ROUTING="dual_task", the two physics
#    paths are task-specific, so PCGrad naturally leaves them untouched and
#    projects only truly shared parameters (primarily the visual backbone).
USE_PCGRAD = False
PCGRAD_EPS = 1e-12

# D. Optional task-specific residual adapters after physics fusion.
# Disabled by default so Phase A/B change no model architecture.
USE_TASK_ADAPTERS = False
TASK_ADAPTER_LEVELS = "deepest"  # all / last2 / deepest
TASK_ADAPTER_BOTTLENECK_RATIO = 0.25
TASK_ADAPTER_SCALE = 1.0

# E. Optional differential learning rates.
USE_DIFFERENTIAL_LR = False
LR_BACKBONE = 2e-4
LR_TASK_HEADS = 3e-4
LR_PHYSICS = 1e-3
LR_ADAPTERS = 5e-4

# F. Optional warm-start stabilization. INIT_MODE below still controls whether
#    Stage-3 weights are loaded. If INIT_MODE="stage3", the visual backbone can
#    be frozen for the first few epochs while new physics modules adapt.
WARM_START_FREEZE_BACKBONE_EPOCHS = 0

# G. Optional exponential moving average for evaluation/checkpoint inference.
USE_EMA = False
EMA_DECAY = 0.999
EMA_UPDATE_AFTER_STEP = 0
USE_EMA_FOR_VALIDATION = True

# =========================================================================
# Physics-informed branch / ablation switches
# =========================================================================

# Master switch. False gives a Stage-3-like path inside this Stage-4 script.
PHYSICS_ENABLED = True

# Task-aware physics routing.
# "shared"            : legacy Stage-4 path; one spatial physics branch feeds
#                       BOTH classification and localization.
# "localization_only" : V2.1 ablation; classification is visual-only and the
#                       spatial physics branch feeds localization only.
# "dual_task"         : V2.2 default; TWO INDEPENDENT physics encoders are used.
#                       Classification receives global late physics modulation,
#                       while localization receives x-aligned spatial fusion.
#                       The two physics parameter sets are disjoint, so their
#                       task gradients cannot interfere inside the physics path.
PHYSICS_TASK_ROUTING = "dual_task"  # shared / localization_only / dual_task

# "vertical"   : reduce image height -> horizontal density profile P(x)
# "horizontal" : reduce image width -> vertical profile; intended mainly as
#                a direction-control ablation, not the primary physics model.
PROFILE_DIRECTION = "vertical"  # vertical / horizontal

# "sum" / "mean" / "weighted_sum"
PROFILE_REDUCTION = "sum"

# weighted_sum uses a fixed center-weighted Gaussian along the reduced axis.
# It is an optional ablation only; the main model uses plain sum.
WEIGHTED_PROFILE_SIGMA_FRACTION = 0.30

# "raw"               : use density profile itself发
# "residual"          : profile - local smooth background
# "raw_plus_residual" : two input channels [raw, residual]
PROFILE_PREPROCESS = "raw"   # 原始是"raw"

# Local smooth background used only by residual modes.
# Must be odd; it is automatically clipped to the available profile length.
PROFILE_SMOOTH_KERNEL = 31

# "sample_zscore" / "minmax" / "none"
# Main model uses sample-wise z-score, independently for each profile channel.
PROFILE_NORMALIZATION = "sample_zscore"
PROFILE_NORM_EPS = 1e-6

# Physics encoder.
PHYSICS_ENCODER = "conv1d"  # conv1d / residual_conv1d
PHYSICS_DIM = 32
PHYSICS_HIDDEN_CHANNELS = (16, 32, 32)

# Fusion.
# gated_residual : F * (1 + alpha*tanh(G))
# add            : F + alpha*G
# concat         : concat(F,G) -> 1x1 Conv2D
# film           : (1 + alpha*tanh(gamma))*F + alpha*tanh(beta)
# none           : no physics fusion (debug/capacity-control switch)
PHYSICS_FUSION = "gated_residual"
PHYSICS_FUSION_LEVELS = "deepest"  # all / last2 / deepest

# Controlled physics-guidance strength.
# bounded   : alpha = PHYSICS_ALPHA_MAX * sigmoid(raw_alpha), recommended
# fixed     : alpha stays exactly PHYSICS_ALPHA_INIT
# unbounded : original V1/V2 learnable alpha; retained only as an ablation
PHYSICS_ALPHA_MODE = "bounded"  # bounded / fixed / unbounded
PHYSICS_ALPHA_INIT = 0.05
PHYSICS_ALPHA_MAX = 0.20
PHYSICS_ALPHA_EPS = 1e-6

# V2.2 classification-specific physics late fusion.
# The classification physics encoder is independent of the localization physics
# encoder. Its 1D feature is globally pooled and then used as a bounded residual
# modulation of the deepest visual feature vector immediately before the FC head.
PHYSICS_CLS_FUSION = "gated_residual"  # gated_residual / add
PHYSICS_CLS_ALPHA_INIT = 0.05
PHYSICS_CLS_ALPHA_MAX = 0.20

# Optional physics-only auxiliary localization heatmap loss.
# Main Stage-4 V1 keeps this OFF to isolate representation/fusion.
USE_PHYSICS_AUX_LOSS = False
LAMBDA_PHYSICS = 0.0

# Initialization.
# "scratch" : recommended paper-primary comparison with Stage 3.
# "stage3"  : optional warm-start engineering experiment.
INIT_MODE = "scratch"  # scratch / stage3
STAGE3_CHECKPOINT = (
    ROOT / "output" / MODEL_NAME
    / "multitask_baseline" / "checkpoint" / "best.pt.tar"
)

# Classification head
SIMPLE_CNN_CLASSIFIER_DROPOUT = 0.20
OTHER_CLASSIFIER_DROPOUT = 0.00

GRAD_CLIP_NORM = 5.0
USE_AMP = True

LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6

EARLY_STOP_PATIENCE = 15
MIN_DELTA = 1e-4
RESUME = False
CHECK_FILE_EXISTS = True

# Stage-2 localization protocol: intentionally shared
IMAGE_WIDTH = detector_base.IMAGE_WIDTH
NUM_BINS = detector_base.NUM_BINS
BIN_WIDTH = detector_base.BIN_WIDTH
GAUSSIAN_SIGMA_BINS = detector_base.GAUSSIAN_SIGMA_BINS
GAUSSIAN_RADIUS_BINS = detector_base.GAUSSIAN_RADIUS_BINS
PRIMARY_TOLERANCE_PX = detector_base.PRIMARY_TOLERANCE_PX
VAL_CHECKPOINT_THRESHOLD = detector_base.VAL_CHECKPOINT_THRESHOLD
THRESHOLD_SEARCH_VALUES = np.asarray(
    detector_base.THRESHOLD_SEARCH_VALUES, dtype=float
)

# Stage-4 V2.3-VAL validation protocol.
# Training supervision is UNCHANGED: class-2 remains masked from localization
# loss. Manual class-2 positions are used ONLY for validation metrics,
# checkpoint/LR/early-stop selection, and validation-only threshold calibration.
REQUIRE_CLASS2_VAL_ANNOTATIONS = True
VAL_MANUAL_STATUS_COLUMN = "manual_loc_status"
VAL_MANUAL_POSITIONS_COLUMN = "manual_excitation_positions"
VAL_LOC_INCLUDE_COLUMN = "loc_eval_include"
VAL_LOC_ELIGIBLE_CLASS2_STATUSES = {"valid", "no_reliable_soliton"}
VALIDATION_PROTOCOL_VERSION = "v2.3_val_aware"

OUTPUT_ROOT = ROOT / "output" / MODEL_NAME / "physics_multitask"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoint"
LOG_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

BEST_CKPT = CHECKPOINT_DIR / "best.pt.tar"
LAST_CKPT = CHECKPOINT_DIR / "last.pt.tar"
LOG_FILE = LOG_DIR / "trainer.log"

# Detailed training diagnostics.
BATCH_LOSS_CSV = METRICS_DIR / "batch_loss_history.csv"
FUSION_ALPHA_HISTORY_CSV = METRICS_DIR / "fusion_alpha_history.csv"
FUSION_ALPHA_FIGURE = FIGURE_DIR / "physics_fusion_alpha_curves.png"
GRAD_DIAGNOSTICS_CSV = METRICS_DIR / "gradient_diagnostics.csv"
GRAD_DIAGNOSTICS_EPOCH_CSV = METRICS_DIR / "gradient_diagnostics_epoch.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. Utilities
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dirs():
    for p in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, FIGURE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def build_logger():
    logger = logging.getLogger("dark_soliton_multitask")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def harmonic_joint_score(classification_macro_f1, localization_f1):
    a = float(classification_macro_f1)
    b = float(localization_f1)
    return 0.0 if (a + b) <= 0 else float(2.0 * a * b / (a + b))


def _safe_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_position_list(value):
    """Parse JSON/Python-list multi-target x positions."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        value_text = str(value).strip()
        if not value_text or value_text.lower() == "nan":
            return []
        try:
            raw = json.loads(value_text)
        except Exception:
            raw = ast.literal_eval(value_text)

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"Localization positions must be a list: {value!r}")

    positions = sorted(float(x) for x in raw)
    for x in positions:
        if not np.isfinite(x) or not (0.0 <= x <= IMAGE_WIDTH - 1):
            raise ValueError(f"Validation position outside [0,{IMAGE_WIDTH - 1}]: {x}")
    return positions


def validate_val_v2_metadata(val_dataset):
    """
    Fail fast if val_v2.csv is incomplete or inconsistent.

    This validates only metadata. It does NOT alter training supervision.
    """
    df = val_dataset.df
    required = {
        VAL_MANUAL_STATUS_COLUMN,
        VAL_MANUAL_POSITIONS_COLUMN,
        VAL_LOC_INCLUDE_COLUMN,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "val_v2.csv is missing manual localization columns: "
            f"{sorted(missing)}. Run make_val_v2.py first."
        )

    label_col = "training_label" if "training_label" in df.columns else "label_v3"
    labels = pd.to_numeric(df[label_col], errors="raise").astype(int)
    class2_indices = np.flatnonzero(labels.to_numpy() == 2)
    if len(class2_indices) == 0:
        raise RuntimeError("val_v2.csv contains no class-2 samples.")

    eligible = 0
    valid_positive = 0
    no_reliable = 0
    ambiguous_or_unusable = 0
    total_gt = 0

    for idx in class2_indices.tolist():
        row = df.iloc[idx]
        status = str(row.get(VAL_MANUAL_STATUS_COLUMN, "")).strip().lower()
        include = _safe_bool(row.get(VAL_LOC_INCLUDE_COLUMN, False))
        positions = parse_position_list(row.get(VAL_MANUAL_POSITIONS_COLUMN, ""))

        if status == "valid":
            if not include or len(positions) == 0:
                raise ValueError(
                    f"{row['file_name']}: class-2 valid sample must be included and have >=1 GT."
                )
            eligible += 1
            valid_positive += 1
            total_gt += len(positions)
        elif status == "no_reliable_soliton":
            if not include or len(positions) != 0:
                raise ValueError(
                    f"{row['file_name']}: no_reliable_soliton must be included with [] GT."
                )
            eligible += 1
            no_reliable += 1
        elif status in {"ambiguous", "unusable"}:
            if include:
                raise ValueError(
                    f"{row['file_name']}: {status} must have loc_eval_include=False."
                )
            ambiguous_or_unusable += 1
        else:
            raise ValueError(
                f"{row['file_name']}: unsupported class-2 manual status={status!r}"
            )

    if REQUIRE_CLASS2_VAL_ANNOTATIONS and eligible == 0:
        raise RuntimeError("No eligible manually annotated class-2 validation samples.")

    return {
        "class2_total": int(len(class2_indices)),
        "class2_eligible": int(eligible),
        "class2_valid_positive": int(valid_positive),
        "class2_no_reliable": int(no_reliable),
        "class2_excluded": int(ambiguous_or_unusable),
        "class2_total_gt": int(total_gt),
    }


def match_positions_one_to_one(gt_positions, pred_positions, tolerance_px):
    """1-D one-to-one matching: maximize matches, then minimize total error."""
    gt = tuple(sorted(float(x) for x in gt_positions))
    pred = tuple(sorted(float(x) for x in pred_positions))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(i, j):
        if i >= len(gt) or j >= len(pred):
            return 0, 0.0, ()

        candidates = [solve(i + 1, j), solve(i, j + 1)]
        error = abs(gt[i] - pred[j])
        if error <= float(tolerance_px):
            sub_count, sub_error, sub_pairs = solve(i + 1, j + 1)
            candidates.append(
                (
                    sub_count + 1,
                    sub_error + error,
                    ((gt[i], pred[j], error),) + sub_pairs,
                )
            )

        return sorted(candidates, key=lambda item: (-item[0], item[1]))[0]

    _, _, pairs = solve(0, 0)
    return [
        {"gt": float(g), "pred": float(p), "error": float(e)}
        for g, p, e in pairs
    ]


def decode_multitarget_localization_cache(localization_cache, threshold):
    records = []
    for item in localization_cache:
        heatmap_probs = item["heatmap_probabilities"]
        offset_probs = item["offset_probabilities"]

        peak_bins = detector_base.find_local_peak_bins(
            probabilities=heatmap_probs,
            threshold=float(threshold),
        )
        pred_positions = [
            detector_base.continuous_position_from_bin(
                bin_index=b,
                offset_probability=offset_probs[b],
            )
            for b in peak_bins
        ]
        pred_scores = [float(heatmap_probs[b]) for b in peak_bins]

        records.append({
            "file_name": item["file_name"],
            "label": int(item["label"]),
            "gt_positions": list(item["gt_positions"]),
            "manual_status": item.get("manual_status", ""),
            "pred_positions": pred_positions,
            "pred_scores": pred_scores,
        })
    return records


def multitarget_localization_metrics_from_cache(localization_cache, threshold):
    records = decode_multitarget_localization_cache(localization_cache, threshold)
    if not records:
        return None, records

    total_gt = 0
    total_pred = 0
    total_tp = 0
    matched_errors = []
    exact_count_hits = 0
    exact_sample_hits = 0
    count_abs_errors = []

    for record in records:
        gt_positions = record["gt_positions"]
        pred_positions = record["pred_positions"]
        pairs = match_positions_one_to_one(
            gt_positions, pred_positions, PRIMARY_TOLERANCE_PX
        )
        n_gt = len(gt_positions)
        n_pred = len(pred_positions)
        n_tp = len(pairs)

        total_gt += n_gt
        total_pred += n_pred
        total_tp += n_tp
        matched_errors.extend(pair["error"] for pair in pairs)

        exact_count = n_pred == n_gt
        exact_sample = exact_count and n_tp == n_gt
        exact_count_hits += int(exact_count)
        exact_sample_hits += int(exact_sample)
        count_abs_errors.append(abs(n_pred - n_gt))

    fp = total_pred - total_tp
    fn = total_gt - total_tp
    precision = total_tp / (total_tp + fp) if (total_tp + fp) > 0 else 0.0
    recall = total_tp / (total_tp + fn) if (total_tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def match_rate(tolerance):
        denominator = sum(len(r["gt_positions"]) for r in records)
        if denominator == 0:
            return 0.0
        matched = sum(
            len(match_positions_one_to_one(r["gt_positions"], r["pred_positions"], tolerance))
            for r in records
        )
        return float(matched / denominator)

    metrics = {
        "num_images": int(len(records)),
        "total_gt_excitations": int(total_gt),
        "total_predictions": int(total_pred),
        "true_positives": int(total_tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision_at_2px": float(precision),
        "recall_at_2px": float(recall),
        "f1_at_2px": float(f1),
        "matched_mae_px": (
            float(np.mean(matched_errors)) if matched_errors else float("nan")
        ),
        "exact_count_accuracy": float(exact_count_hits / len(records)),
        "exact_sample_success_at_2px": float(exact_sample_hits / len(records)),
        "count_mae": float(np.mean(count_abs_errors)),
        "false_positives_per_image": float(fp / len(records)),
        "false_negatives_per_image": float(fn / len(records)),
        "mean_predictions_per_image": float(total_pred / len(records)),
        "gt_match_rate_at_1px": match_rate(1.0),
        "gt_match_rate_at_2px": match_rate(2.0),
        "gt_match_rate_at_3px": match_rate(3.0),
    }
    return metrics, records


def combine_localization_scores(loc01_metrics, loc_c2_metrics):
    """Return task-balanced LocScore and descriptive combined micro metrics."""
    if loc_c2_metrics is None:
        return {
            "loc_score": float(loc01_metrics["f1_at_2px"]),
            "combined_precision_at_2px": float(loc01_metrics["precision_at_2px"]),
            "combined_recall_at_2px": float(loc01_metrics["recall_at_2px"]),
            "combined_micro_f1_at_2px": float(loc01_metrics["f1_at_2px"]),
            "combined_false_positives_per_image": float(
                loc01_metrics["false_positives_per_image"]
            ),
            "combined_matched_mae_px": float(loc01_metrics["matched_mae_px"]),
        }

    loc_score = harmonic_joint_score(
        loc01_metrics["f1_at_2px"], loc_c2_metrics["f1_at_2px"]
    )
    tp = int(loc01_metrics["true_positives"]) + int(loc_c2_metrics["true_positives"])
    fp = int(loc01_metrics["false_positives"]) + int(loc_c2_metrics["false_positives"])
    fn = int(loc01_metrics["false_negatives"]) + int(loc_c2_metrics["false_negatives"])
    n_img = int(loc01_metrics.get("positive_samples", 0))  # overwritten below
    # loc01's records include class-0 negatives, so infer count from FP/image if
    # possible is unsafe. Caller stores the exact number separately; use the
    # known val split record count fields added below when available.
    n_img = int(loc01_metrics.get("num_images", 0)) + int(loc_c2_metrics["num_images"])
    if int(loc01_metrics.get("num_images", 0)) == 0:
        # Old strict metric has no num_images key; caller may fill it.
        n_img = int(loc_c2_metrics["num_images"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mae_num = 0.0
    mae_den = 0
    for metrics in (loc01_metrics, loc_c2_metrics):
        count = int(metrics["true_positives"])
        mae = float(metrics["matched_mae_px"])
        if count > 0 and np.isfinite(mae):
            mae_num += mae * count
            mae_den += count

    total_images = int(loc01_metrics.get("record_count", 0)) + int(loc_c2_metrics["num_images"])
    fp_per_image = float(fp / total_images) if total_images > 0 else 0.0

    return {
        "loc_score": float(loc_score),
        "combined_precision_at_2px": float(precision),
        "combined_recall_at_2px": float(recall),
        "combined_micro_f1_at_2px": float(micro_f1),
        "combined_false_positives_per_image": fp_per_image,
        "combined_matched_mae_px": (
            float(mae_num / mae_den) if mae_den > 0 else float("nan")
        ),
    }


# =============================================================================
# 3. Physics-informed model
# =============================================================================

def validate_physics_configuration():
    valid_direction = {"vertical", "horizontal"}
    valid_reduction = {"sum", "mean", "weighted_sum"}
    valid_preprocess = {"raw", "residual", "raw_plus_residual"}
    valid_norm = {"sample_zscore", "minmax", "none"}
    valid_encoder = {"conv1d", "residual_conv1d"}
    valid_fusion = {"gated_residual", "add", "concat", "film", "none"}
    valid_levels = {"all", "last2", "deepest"}
    valid_init = {"scratch", "stage3"}
    valid_grad_scope = {"backbone", "physics_cls", "physics_loc", "physics", "shared"}
    valid_alpha_mode = {"bounded", "fixed", "unbounded"}
    valid_task_routing = {"shared", "localization_only", "dual_task"}
    valid_cls_fusion = {"gated_residual", "add"}

    checks = [
        (PROFILE_DIRECTION, valid_direction, "PROFILE_DIRECTION"),
        (PROFILE_REDUCTION, valid_reduction, "PROFILE_REDUCTION"),
        (PROFILE_PREPROCESS, valid_preprocess, "PROFILE_PREPROCESS"),
        (PROFILE_NORMALIZATION, valid_norm, "PROFILE_NORMALIZATION"),
        (PHYSICS_ENCODER, valid_encoder, "PHYSICS_ENCODER"),
        (PHYSICS_FUSION, valid_fusion, "PHYSICS_FUSION"),
        (PHYSICS_FUSION_LEVELS, valid_levels, "PHYSICS_FUSION_LEVELS"),
        (TASK_ADAPTER_LEVELS, valid_levels, "TASK_ADAPTER_LEVELS"),
        (INIT_MODE, valid_init, "INIT_MODE"),
        (PHYSICS_ALPHA_MODE, valid_alpha_mode, "PHYSICS_ALPHA_MODE"),
        (PHYSICS_TASK_ROUTING, valid_task_routing, "PHYSICS_TASK_ROUTING"),
        (PHYSICS_CLS_FUSION, valid_cls_fusion, "PHYSICS_CLS_FUSION"),
    ]

    for value, valid, name in checks:
        if value not in valid:
            raise ValueError(
                f"{name}={value!r} is invalid. "
                f"Choose from {sorted(valid)}."
            )

    if not GRAD_DIAGNOSTIC_SCOPES:
        raise ValueError("GRAD_DIAGNOSTIC_SCOPES must not be empty.")
    for scope in GRAD_DIAGNOSTIC_SCOPES:
        if scope not in valid_grad_scope:
            raise ValueError(
                f"Invalid gradient diagnostic scope {scope!r}; "
                f"choose from {sorted(valid_grad_scope)}."
            )

    if float(PHYSICS_ALPHA_INIT) < 0.0:
        raise ValueError("PHYSICS_ALPHA_INIT must be >= 0.")
    if float(PHYSICS_ALPHA_MAX) <= 0.0:
        raise ValueError("PHYSICS_ALPHA_MAX must be > 0.")
    if PHYSICS_ALPHA_MODE == "bounded":
        if not (0.0 < float(PHYSICS_ALPHA_INIT) < float(PHYSICS_ALPHA_MAX)):
            raise ValueError(
                "For bounded alpha, require 0 < PHYSICS_ALPHA_INIT "
                "< PHYSICS_ALPHA_MAX."
            )
    if not (0.0 < float(PHYSICS_ALPHA_EPS) < 0.5):
        raise ValueError("PHYSICS_ALPHA_EPS must lie in (0, 0.5).")

    if float(PHYSICS_CLS_ALPHA_INIT) < 0.0:
        raise ValueError("PHYSICS_CLS_ALPHA_INIT must be >= 0.")
    if float(PHYSICS_CLS_ALPHA_MAX) <= 0.0:
        raise ValueError("PHYSICS_CLS_ALPHA_MAX must be > 0.")
    if PHYSICS_ALPHA_MODE == "bounded":
        if not (0.0 < float(PHYSICS_CLS_ALPHA_INIT) < float(PHYSICS_CLS_ALPHA_MAX)):
            raise ValueError(
                "For bounded classification alpha, require 0 < "
                "PHYSICS_CLS_ALPHA_INIT < PHYSICS_CLS_ALPHA_MAX."
            )

    if PHYSICS_DIM <= 0:
        raise ValueError("PHYSICS_DIM must be positive.")

    if len(PHYSICS_HIDDEN_CHANNELS) == 0:
        raise ValueError(
            "PHYSICS_HIDDEN_CHANNELS must contain at least one channel."
        )

    if any(int(c) <= 0 for c in PHYSICS_HIDDEN_CHANNELS):
        raise ValueError(
            "All PHYSICS_HIDDEN_CHANNELS must be positive."
        )

    if not (0.0 < float(WEIGHTED_PROFILE_SIGMA_FRACTION) <= 1.0):
        raise ValueError(
            "WEIGHTED_PROFILE_SIGMA_FRACTION must be in (0, 1]."
        )

    if int(PROFILE_SMOOTH_KERNEL) < 3:
        raise ValueError(
            "PROFILE_SMOOTH_KERNEL must be >= 3."
        )

    if float(PROFILE_NORM_EPS) <= 0:
        raise ValueError(
            "PROFILE_NORM_EPS must be positive."
        )

    if USE_PHYSICS_AUX_LOSS and not PHYSICS_ENABLED:
        raise ValueError(
            "USE_PHYSICS_AUX_LOSS=True requires PHYSICS_ENABLED=True."
        )

    if USE_PHYSICS_AUX_LOSS and float(LAMBDA_PHYSICS) <= 0:
        raise ValueError(
            "USE_PHYSICS_AUX_LOSS=True requires LAMBDA_PHYSICS > 0."
        )

    if float(LAMBDA_HEATMAP) < 0:
        raise ValueError("LAMBDA_HEATMAP must be non-negative.")

    if int(GRAD_DIAGNOSTIC_INTERVAL) <= 0:
        raise ValueError("GRAD_DIAGNOSTIC_INTERVAL must be >= 1.")

    if not (0.0 < float(TASK_ADAPTER_BOTTLENECK_RATIO) <= 1.0):
        raise ValueError(
            "TASK_ADAPTER_BOTTLENECK_RATIO must be in (0, 1]."
        )

    if float(TASK_ADAPTER_SCALE) < 0:
        raise ValueError("TASK_ADAPTER_SCALE must be non-negative.")

    if int(WARM_START_FREEZE_BACKBONE_EPOCHS) < 0:
        raise ValueError(
            "WARM_START_FREEZE_BACKBONE_EPOCHS must be >= 0."
        )

    if USE_EMA and not (0.0 < float(EMA_DECAY) < 1.0):
        raise ValueError("EMA_DECAY must be in (0, 1).")


class ClassificationHead(nn.Module):
    """Deepest 2D feature -> global feature -> 3-class logits.

    V2.2 exposes feature extraction and classification separately so an
    independent global physics cue can be fused immediately before the FC head.
    """

    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_channels, num_classes)

    def extract_feature(self, deepest_feature):
        feature = self.pool(deepest_feature)
        return torch.flatten(feature, 1)

    def classify_feature(self, feature):
        return self.fc(self.dropout(feature))

    def forward(self, deepest_feature):
        feature = self.extract_feature(deepest_feature)
        logits = self.classify_feature(feature)
        return logits, feature


def _make_center_weights(length, device, dtype):
    """
    Fixed smooth center weighting for optional weighted_sum profile.
    We normalize weights to mean 1, so the scale remains sum-like.
    """
    if length <= 1:
        return torch.ones(
            length,
            device=device,
            dtype=dtype,
        )

    coords = torch.linspace(
        -1.0,
        1.0,
        steps=length,
        device=device,
        dtype=dtype,
    )

    sigma = float(
        WEIGHTED_PROFILE_SIGMA_FRACTION
    )

    weights = torch.exp(
        -0.5 * (coords / sigma) ** 2
    )

    weights = weights / (
        weights.mean() + PROFILE_NORM_EPS
    )

    return weights


def _safe_smooth_1d(profile):
    """
    Moving-average background for the optional residual profile modes.
    Input: [B, C, L]
    Output: [B, C, L]
    """
    length = int(profile.shape[-1])

    if length < 3:
        return profile

    kernel = min(
        int(PROFILE_SMOOTH_KERNEL),
        length - 1 if length % 2 == 0 else length,
    )

    if kernel % 2 == 0:
        kernel -= 1

    kernel = max(kernel, 3)

    pad = kernel // 2

    padded = F.pad(
        profile,
        (pad, pad),
        mode="reflect",
    )

    return F.avg_pool1d(
        padded,
        kernel_size=kernel,
        stride=1,
        padding=0,
    )


def build_physics_profile(images):
    """
    Build a deterministic 1D representation directly from the normalized image.

    images: [B, 1, H, W]

    vertical:
        reduce H -> [B,1,W], preserving horizontal coordinate x.

    horizontal:
        reduce W -> [B,1,H]. This option is mainly a direction-control
        ablation. Its sequence is still resampled to each feature-map width
        by the fusion modules so network capacity stays comparable.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected images [B,1,H,W], got {tuple(images.shape)}"
        )

    if PROFILE_DIRECTION == "vertical":
        reduction_dim = 2
        reduction_length = int(images.shape[2])
    elif PROFILE_DIRECTION == "horizontal":
        reduction_dim = 3
        reduction_length = int(images.shape[3])
    else:
        raise ValueError(
            f"Unsupported PROFILE_DIRECTION={PROFILE_DIRECTION!r}"
        )

    if PROFILE_REDUCTION == "sum":
        profile = images.sum(
            dim=reduction_dim
        )

    elif PROFILE_REDUCTION == "mean":
        profile = images.mean(
            dim=reduction_dim
        )

    elif PROFILE_REDUCTION == "weighted_sum":
        weights = _make_center_weights(
            reduction_length,
            device=images.device,
            dtype=images.dtype,
        )

        if reduction_dim == 2:
            weights = weights.view(
                1, 1, reduction_length, 1
            )
        else:
            weights = weights.view(
                1, 1, 1, reduction_length
            )

        profile = (
            images * weights
        ).sum(
            dim=reduction_dim
        )

    else:
        raise ValueError(
            f"Unsupported PROFILE_REDUCTION={PROFILE_REDUCTION!r}"
        )

    if profile.ndim != 3:
        raise RuntimeError(
            f"Physics profile must be [B,C,L], got {tuple(profile.shape)}"
        )

    if PROFILE_PREPROCESS == "raw":
        physics_input = profile

    elif PROFILE_PREPROCESS == "residual":
        background = _safe_smooth_1d(
            profile
        )
        physics_input = profile - background

    elif PROFILE_PREPROCESS == "raw_plus_residual":
        background = _safe_smooth_1d(
            profile
        )
        residual = profile - background
        physics_input = torch.cat(
            [profile, residual],
            dim=1,
        )

    else:
        raise ValueError(
            f"Unsupported PROFILE_PREPROCESS={PROFILE_PREPROCESS!r}"
        )

    if PROFILE_NORMALIZATION == "sample_zscore":
        mean = physics_input.mean(
            dim=-1,
            keepdim=True,
        )
        std = physics_input.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )
        physics_input = (
            physics_input - mean
        ) / (
            std + PROFILE_NORM_EPS
        )

    elif PROFILE_NORMALIZATION == "minmax":
        p_min = physics_input.amin(
            dim=-1,
            keepdim=True,
        )
        p_max = physics_input.amax(
            dim=-1,
            keepdim=True,
        )
        physics_input = (
            2.0
            * (physics_input - p_min)
            / (p_max - p_min + PROFILE_NORM_EPS)
            - 1.0
        )

    elif PROFILE_NORMALIZATION == "none":
        pass

    else:
        raise ValueError(
            f"Unsupported PROFILE_NORMALIZATION={PROFILE_NORMALIZATION!r}"
        )

    return physics_input


class ConvBNAct1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(
            channels
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(
            channels
        )
        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        residual = x

        x = self.act(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.bn2(
            self.conv2(x)
        )

        return self.act(
            x + residual
        )


class PhysicsEncoder1D(nn.Module):
    """
    Lightweight 1D encoder that preserves sequence length.

    Default conv1d:
        in -> 16 -> 32 -> 32(PHYSICS_DIM)

    residual_conv1d:
        stem -> PHYSICS_DIM -> residual blocks
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder_type,
    ):
        super().__init__()

        hidden = [
            int(c)
            for c in PHYSICS_HIDDEN_CHANNELS
        ]

        if encoder_type == "conv1d":
            channels = (
                [int(in_channels)]
                + hidden
            )

            blocks = []

            kernels = [7, 5, 3]

            for i in range(
                len(channels) - 1
            ):
                kernel = kernels[
                    min(
                        i,
                        len(kernels) - 1,
                    )
                ]

                blocks.append(
                    ConvBNAct1D(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernel,
                    )
                )

            if channels[-1] != int(out_channels):
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            channels[-1],
                            int(out_channels),
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.BatchNorm1d(
                            int(out_channels)
                        ),
                        nn.ReLU(
                            inplace=True
                        ),
                    )
                )

            self.net = nn.Sequential(
                *blocks
            )

        elif encoder_type == "residual_conv1d":
            stem_channels = int(
                out_channels
            )

            self.net = nn.Sequential(
                ConvBNAct1D(
                    int(in_channels),
                    stem_channels,
                    kernel_size=7,
                ),
                ResidualBlock1D(
                    stem_channels,
                    kernel_size=5,
                ),
                ResidualBlock1D(
                    stem_channels,
                    kernel_size=3,
                ),
            )

        else:
            raise ValueError(
                f"Unsupported PHYSICS_ENCODER={encoder_type!r}"
            )

    def forward(self, x):
        return self.net(x)


def fusion_level_indices(num_levels):
    if num_levels <= 0:
        raise ValueError(
            "Backbone must expose at least one feature level."
        )

    if PHYSICS_FUSION_LEVELS == "all":
        return set(
            range(num_levels)
        )

    if PHYSICS_FUSION_LEVELS == "last2":
        start = max(
            0,
            num_levels - 2,
        )
        return set(
            range(start, num_levels)
        )

    if PHYSICS_FUSION_LEVELS == "deepest":
        return {
            num_levels - 1
        }

    raise ValueError(
        f"Unsupported PHYSICS_FUSION_LEVELS="
        f"{PHYSICS_FUSION_LEVELS!r}"
    )


class PhysicsFusionLevel(nn.Module):
    """
    Fuse one 1D physics feature sequence into one 2D feature map.

    Physics features are interpolated to the 2D feature-map width and then
    broadcast along height, preserving x-aligned guidance for the main
    vertical-profile setting.
    """

    def __init__(
        self,
        physics_channels,
        feature_channels,
        fusion_type,
    ):
        super().__init__()

        self.feature_channels = int(
            feature_channels
        )
        self.fusion_type = str(
            fusion_type
        )

        if self.fusion_type in {
            "gated_residual",
            "add",
            "concat",
        }:
            projected_channels = (
                self.feature_channels
            )

        elif self.fusion_type == "film":
            projected_channels = (
                2 * self.feature_channels
            )

        elif self.fusion_type == "none":
            projected_channels = 0

        else:
            raise ValueError(
                f"Unsupported fusion type: "
                f"{self.fusion_type!r}"
            )

        if projected_channels > 0:
            self.physics_projection = (
                nn.Sequential(
                    nn.Conv1d(
                        int(physics_channels),
                        projected_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(
                        projected_channels
                    ),
                )
            )
        else:
            self.physics_projection = None

        if self.fusion_type == "concat":
            self.concat_projection = (
                nn.Sequential(
                    nn.Conv2d(
                        2 * self.feature_channels,
                        self.feature_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        self.feature_channels
                    ),
                    nn.ReLU(
                        inplace=True
                    ),
                )
            )
        else:
            self.concat_projection = None

        self.alpha_mode = str(PHYSICS_ALPHA_MODE)
        self.alpha_max = float(PHYSICS_ALPHA_MAX)

        if self.fusion_type in {
            "gated_residual",
            "add",
            "film",
        }:
            if self.alpha_mode == "bounded":
                # Keep the historical state-dict key "alpha". For bounded mode
                # it stores an unconstrained logit; effective alpha is bounded.
                ratio = float(PHYSICS_ALPHA_INIT) / self.alpha_max
                ratio = min(
                    max(ratio, float(PHYSICS_ALPHA_EPS)),
                    1.0 - float(PHYSICS_ALPHA_EPS),
                )
                raw_alpha = math.log(ratio / (1.0 - ratio))
                self.alpha = nn.Parameter(
                    torch.tensor(raw_alpha, dtype=torch.float32)
                )
            elif self.alpha_mode == "unbounded":
                self.alpha = nn.Parameter(
                    torch.tensor(float(PHYSICS_ALPHA_INIT), dtype=torch.float32)
                )
            elif self.alpha_mode == "fixed":
                self.register_buffer(
                    "alpha",
                    torch.tensor(float(PHYSICS_ALPHA_INIT), dtype=torch.float32),
                )
            else:
                raise ValueError(
                    f"Unsupported PHYSICS_ALPHA_MODE={self.alpha_mode!r}"
                )
        else:
            self.register_parameter("alpha", None)

    def effective_alpha(self):
        """Actual fusion strength used by forward()."""
        if self.alpha is None:
            return None
        if self.alpha_mode == "bounded":
            return self.alpha_max * torch.sigmoid(self.alpha)
        return self.alpha

    def _align(self, physics_feature, width):
        x = physics_feature

        if x.shape[-1] != int(width):
            x = F.interpolate(
                x,
                size=int(width),
                mode="linear",
                align_corners=False,
            )

        return x

    def forward(
        self,
        feature,
        physics_feature,
    ):
        if self.fusion_type == "none":
            return feature

        if feature.ndim != 4:
            raise ValueError(
                "2D feature must be [B,C,H,W]."
            )

        width = int(
            feature.shape[-1]
        )

        physics = self._align(
            physics_feature,
            width,
        )

        projected = self.physics_projection(
            physics
        )

        if self.fusion_type == "gated_residual":
            gate = torch.tanh(projected).unsqueeze(2)
            alpha = self.effective_alpha()
            return feature * (1.0 + alpha * gate)

        if self.fusion_type == "add":
            addition = projected.unsqueeze(2)
            alpha = self.effective_alpha()
            return feature + alpha * addition

        if self.fusion_type == "concat":
            physics_2d = projected.unsqueeze(
                2
            ).expand(
                -1,
                -1,
                feature.shape[2],
                -1,
            )

            fused = torch.cat(
                [
                    feature,
                    physics_2d,
                ],
                dim=1,
            )

            return self.concat_projection(
                fused
            )

        if self.fusion_type == "film":
            gamma, beta = torch.chunk(
                projected,
                chunks=2,
                dim=1,
            )

            alpha = self.effective_alpha()
            gamma = (alpha * torch.tanh(gamma)).unsqueeze(2)
            beta = (alpha * torch.tanh(beta)).unsqueeze(2)

            return (
                (1.0 + gamma) * feature
                + beta
            )

        raise RuntimeError(
            "Unexpected physics fusion type."
        )


class PhysicsClassificationLateFusion(nn.Module):
    """Global late fusion for the classification-specific physics path.

    Input:
        visual_feature  : [B, C] from GAP of deepest visual feature
        physics_feature : [B, P, L] from the independent cls physics encoder

    The physics sequence is globally averaged, projected to C channels, then
    injected as a bounded residual modulation. This preserves the visual
    classifier as the identity path when alpha is small while allowing the
    density-profile branch to contribute global semantic information.
    """

    def __init__(self, physics_channels, visual_channels, fusion_type):
        super().__init__()
        self.fusion_type = str(fusion_type)
        self.visual_channels = int(visual_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(int(physics_channels), self.visual_channels, bias=False),
            nn.LayerNorm(self.visual_channels),
        )

        self.alpha_mode = str(PHYSICS_ALPHA_MODE)
        self.alpha_max = float(PHYSICS_CLS_ALPHA_MAX)

        if self.alpha_mode == "bounded":
            ratio = float(PHYSICS_CLS_ALPHA_INIT) / self.alpha_max
            ratio = min(
                max(ratio, float(PHYSICS_ALPHA_EPS)),
                1.0 - float(PHYSICS_ALPHA_EPS),
            )
            raw_alpha = math.log(ratio / (1.0 - ratio))
            self.alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        elif self.alpha_mode == "unbounded":
            self.alpha = nn.Parameter(
                torch.tensor(float(PHYSICS_CLS_ALPHA_INIT), dtype=torch.float32)
            )
        elif self.alpha_mode == "fixed":
            self.register_buffer(
                "alpha",
                torch.tensor(float(PHYSICS_CLS_ALPHA_INIT), dtype=torch.float32),
            )
        else:
            raise ValueError(
                f"Unsupported PHYSICS_ALPHA_MODE={self.alpha_mode!r}"
            )

    def effective_alpha(self):
        if self.alpha_mode == "bounded":
            return self.alpha_max * torch.sigmoid(self.alpha)
        return self.alpha

    def forward(self, visual_feature, physics_feature):
        if visual_feature.ndim != 2:
            raise ValueError(
                f"visual_feature must be [B,C], got {tuple(visual_feature.shape)}"
            )
        if physics_feature.ndim != 3:
            raise ValueError(
                f"physics_feature must be [B,P,L], got {tuple(physics_feature.shape)}"
            )

        physics_global = self.pool(physics_feature).squeeze(-1)
        projected = self.projection(physics_global)
        alpha = self.effective_alpha()

        if self.fusion_type == "gated_residual":
            gate = torch.tanh(projected)
            return visual_feature * (1.0 + alpha * gate)

        if self.fusion_type == "add":
            return visual_feature + alpha * torch.tanh(projected)

        raise RuntimeError(
            f"Unexpected PHYSICS_CLS_FUSION={self.fusion_type!r}"
        )


class PhysicsAuxiliaryHead(nn.Module):
    """
    Optional physics-only heatmap head.

    It uses the same class-0/class-1 heatmap targets as the main localization
    branch, but has no offset head. It is OFF in the paper-primary Stage-4 V1.
    """

    def __init__(
        self,
        in_channels,
        num_bins,
    ):
        super().__init__()

        self.num_bins = int(
            num_bins
        )
        self.head = nn.Conv1d(
            int(in_channels),
            1,
            kernel_size=1,
        )

        prior_probability = 0.01
        heatmap_bias = -np.log(
            (1.0 - prior_probability)
            / prior_probability
        )

        nn.init.constant_(
            self.head.bias,
            float(heatmap_bias),
        )

    def forward(
        self,
        physics_feature,
    ):
        x = physics_feature

        if x.shape[-1] != self.num_bins:
            x = F.interpolate(
                x,
                size=self.num_bins,
                mode="linear",
                align_corners=False,
            )

        return self.head(
            x
        ).squeeze(1)


class TaskAdapter2D(nn.Module):
    """
    Lightweight task-specific residual adapter.

    The last projection is zero-initialized, so enabling adapters starts from
    the identity mapping and does not immediately destroy the shared feature.
    """

    def __init__(
        self,
        channels,
        bottleneck_ratio=0.25,
        scale=1.0,
    ):
        super().__init__()

        channels = int(channels)
        hidden = max(4, int(round(channels * float(bottleneck_ratio))))
        self.scale = float(scale)

        self.down = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.kaiming_normal_(
            self.down.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.up(
            self.act(
                self.down(x)
            )
        )
        return x + self.scale * residual


def build_task_adapter_list(out_channels):
    # Adapter selection is intentionally independent of physics-fusion levels.
    if TASK_ADAPTER_LEVELS == "all":
        active_levels = set(range(len(out_channels)))
    elif TASK_ADAPTER_LEVELS == "last2":
        active_levels = set(range(max(0, len(out_channels) - 2), len(out_channels)))
    elif TASK_ADAPTER_LEVELS == "deepest":
        active_levels = {len(out_channels) - 1}
    else:
        raise ValueError(
            f"Unsupported TASK_ADAPTER_LEVELS={TASK_ADAPTER_LEVELS!r}"
        )

    return nn.ModuleList([
        (
            TaskAdapter2D(
                channels=channels,
                bottleneck_ratio=TASK_ADAPTER_BOTTLENECK_RATIO,
                scale=TASK_ADAPTER_SCALE,
            )
            if idx in active_levels
            else nn.Identity()
        )
        for idx, channels in enumerate(out_channels)
    ])


class PhysicsInformedMultiTaskModel(nn.Module):
    """Stage 4 V2.2 task-decoupled dual-physics model.

    Routing modes retained for controlled ablation:
        shared:
            one spatial physics branch -> classification + localization
        localization_only:
            visual-only classification + one spatial physics localization branch
        dual_task (V2.2 default):
            independent cls physics encoder -> global late fusion for cls
            independent loc physics encoder -> x-aligned spatial fusion for loc

    In dual_task mode the two physics parameter sets are disjoint. Therefore
    classification and localization cannot create gradient conflict inside the
    physics encoders/fusions; only the visual backbone remains truly shared.
    """

    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.model_name = model_name
        self.backbone = detector_base.build_feature_backbone(model_name)

        deepest_channels = self.backbone.out_channels[-1]
        dropout = (
            SIMPLE_CNN_CLASSIFIER_DROPOUT
            if model_name.lower().strip() == "simple_cnn"
            else OTHER_CLASSIFIER_DROPOUT
        )
        profile_input_channels = (
            2 if PROFILE_PREPROCESS == "raw_plus_residual" else 1
        )

        # Legacy attribute names are intentionally retained for the LOCALIZATION
        # physics path so existing Stage-4 evaluators can still recover the
        # spatial fusion modules and learned localization alphas.
        if PHYSICS_ENABLED:
            self.physics_encoder = PhysicsEncoder1D(
                in_channels=profile_input_channels,
                out_channels=PHYSICS_DIM,
                encoder_type=PHYSICS_ENCODER,
            )
            active_levels = fusion_level_indices(len(self.backbone.out_channels))
            self.physics_fusions = nn.ModuleList([
                (
                    PhysicsFusionLevel(
                        physics_channels=PHYSICS_DIM,
                        feature_channels=channels,
                        fusion_type=PHYSICS_FUSION,
                    )
                    if idx in active_levels
                    else nn.Identity()
                )
                for idx, channels in enumerate(self.backbone.out_channels)
            ])

            # V2.2: independent classification physics path.
            if PHYSICS_TASK_ROUTING == "dual_task":
                self.physics_encoder_cls = PhysicsEncoder1D(
                    in_channels=profile_input_channels,
                    out_channels=PHYSICS_DIM,
                    encoder_type=PHYSICS_ENCODER,
                )
                self.physics_cls_fusion = PhysicsClassificationLateFusion(
                    physics_channels=PHYSICS_DIM,
                    visual_channels=deepest_channels,
                    fusion_type=PHYSICS_CLS_FUSION,
                )
            else:
                self.physics_encoder_cls = None
                self.physics_cls_fusion = None

            if USE_PHYSICS_AUX_LOSS:
                # Auxiliary heatmap supervision belongs to the localization
                # physics path only.
                self.physics_aux_head = PhysicsAuxiliaryHead(
                    in_channels=PHYSICS_DIM,
                    num_bins=NUM_BINS,
                )
            else:
                self.physics_aux_head = None
        else:
            self.physics_encoder = None
            self.physics_fusions = None
            self.physics_encoder_cls = None
            self.physics_cls_fusion = None
            self.physics_aux_head = None

        if USE_TASK_ADAPTERS:
            self.classification_adapters = build_task_adapter_list(
                self.backbone.out_channels
            )
            self.localization_adapters = build_task_adapter_list(
                self.backbone.out_channels
            )
        else:
            self.classification_adapters = None
            self.localization_adapters = None

        self.classification_head = ClassificationHead(
            in_channels=deepest_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.localization_head = detector_base.MultiScaleLocalizationHead(
            in_channels_list=self.backbone.out_channels,
            num_bins=NUM_BINS,
            fusion_dim=detector_base.FUSION_DIM,
            hidden_dim=detector_base.HEAD_HIDDEN_DIM,
            dropout=detector_base.HEAD_DROPOUT,
        )

    def _fuse_features(self, features, physics_feature):
        if not PHYSICS_ENABLED:
            return list(features)
        fused = []
        for feature, fusion_module in zip(features, self.physics_fusions):
            if isinstance(fusion_module, nn.Identity):
                fused.append(feature)
            else:
                fused.append(fusion_module(feature, physics_feature))
        return fused

    def _apply_task_adapters(self, features, adapters):
        if adapters is None:
            return list(features)
        return [
            adapter(feature)
            for feature, adapter in zip(features, adapters)
        ]

    def forward(self, images):
        features = self.backbone(images)

        physics_profile = None
        physics_feature_loc = None
        physics_feature_cls = None
        physics_aux_heatmap_logits = None

        if PHYSICS_ENABLED:
            # Build the deterministic profile once; the two task-specific
            # encoders learn independent representations from the same physics
            # source in dual_task mode.
            physics_profile = build_physics_profile(images)
            physics_feature_loc = self.physics_encoder(physics_profile)
            localization_fused_features = self._fuse_features(
                features, physics_feature_loc
            )

            if PHYSICS_TASK_ROUTING == "dual_task":
                physics_feature_cls = self.physics_encoder_cls(physics_profile)

            if self.physics_aux_head is not None:
                physics_aux_heatmap_logits = self.physics_aux_head(
                    physics_feature_loc
                )
        else:
            localization_fused_features = list(features)

        if PHYSICS_TASK_ROUTING == "shared":
            classification_base_features = localization_fused_features
            localization_base_features = localization_fused_features
        elif PHYSICS_TASK_ROUTING in {"localization_only", "dual_task"}:
            classification_base_features = list(features)
            localization_base_features = localization_fused_features
        else:
            raise RuntimeError(
                f"Unexpected PHYSICS_TASK_ROUTING={PHYSICS_TASK_ROUTING!r}"
            )

        classification_features = self._apply_task_adapters(
            classification_base_features,
            self.classification_adapters,
        )
        localization_features = self._apply_task_adapters(
            localization_base_features,
            self.localization_adapters,
        )

        if PHYSICS_ENABLED and PHYSICS_TASK_ROUTING == "dual_task":
            visual_cls_feature = self.classification_head.extract_feature(
                classification_features[-1]
            )
            classification_feature = self.physics_cls_fusion(
                visual_cls_feature,
                physics_feature_cls,
            )
            classification_logits = self.classification_head.classify_feature(
                classification_feature
            )
            shared_feature = classification_feature
        else:
            classification_logits, shared_feature = self.classification_head(
                classification_features[-1]
            )

        localization = self.localization_head(localization_features)

        return {
            "classification_logits": classification_logits,
            "shared_feature": shared_feature,
            "heatmap_logits": localization["heatmap_logits"],
            "offset_logits": localization["offset_logits"],
            "physics_profile": physics_profile,
            # Legacy alias = localization physics feature for evaluator
            # compatibility.
            "physics_feature": physics_feature_loc,
            "physics_feature_loc": physics_feature_loc,
            "physics_feature_cls": physics_feature_cls,
            "physics_aux_heatmap_logits": physics_aux_heatmap_logits,
        }


def get_fusion_alpha_values(model):
    """Return effective/raw alpha values for localization and cls physics paths."""
    values = {}

    # Localization spatial fusion (legacy key names retained).
    fusion_modules = getattr(model, "physics_fusions", None)
    if fusion_modules is not None:
        for level_index, module in enumerate(fusion_modules):
            alpha = getattr(module, "alpha", None)
            key = f"physics_alpha_level{level_index}"
            raw_key = f"physics_alpha_rawparam_level{level_index}"
            if alpha is None:
                values[key] = float("nan")
                values[raw_key] = float("nan")
            else:
                effective_fn = getattr(module, "effective_alpha", None)
                effective_alpha = effective_fn() if callable(effective_fn) else alpha
                values[key] = float(effective_alpha.detach().cpu().item())
                values[raw_key] = float(alpha.detach().cpu().item())

    # Classification global late fusion.
    cls_fusion = getattr(model, "physics_cls_fusion", None)
    if cls_fusion is not None:
        alpha = getattr(cls_fusion, "alpha", None)
        if alpha is not None:
            effective_fn = getattr(cls_fusion, "effective_alpha", None)
            effective_alpha = effective_fn() if callable(effective_fn) else alpha
            values["physics_cls_alpha"] = float(
                effective_alpha.detach().cpu().item()
            )
            values["physics_cls_alpha_rawparam"] = float(
                alpha.detach().cpu().item()
            )
        else:
            values["physics_cls_alpha"] = float("nan")
            values["physics_cls_alpha_rawparam"] = float("nan")

    return values


def save_fusion_alpha_history(history_df):
    """
    Save epoch-wise learned fusion alpha values and a dedicated curve figure.
    """
    alpha_columns = [
        column
        for column in history_df.columns
        if (
            column.startswith("physics_alpha_level")
            or column == "physics_cls_alpha"
        )
    ]

    if not alpha_columns:
        return

    alpha_df = history_df[
        ["epoch"] + alpha_columns
    ].copy()

    alpha_df.to_csv(
        FUSION_ALPHA_HISTORY_CSV,
        index=False,
    )

    usable_columns = [
        column
        for column in alpha_columns
        if not alpha_df[
            column
        ].isna().all()
    ]

    if not usable_columns:
        return

    fig = plt.figure(
        figsize=(6.6, 4.8)
    )

    for column in usable_columns:
        level_name = (
            "Cls late fusion"
            if column == "physics_cls_alpha"
            else column.replace("physics_alpha_level", "Loc level ")
        )

        plt.plot(
            alpha_df["epoch"],
            alpha_df[column],
            marker="o",
            linewidth=1.4,
            markersize=3.5,
            label=level_name,
        )

    plt.axhline(
        y=float(PHYSICS_ALPHA_INIT),
        linestyle="--",
        linewidth=1.0,
        label="Initial alpha",
    )
    if PHYSICS_ALPHA_MODE == "bounded":
        plt.axhline(
            y=float(PHYSICS_ALPHA_MAX),
            linestyle=":",
            linewidth=1.0,
            label="Alpha upper bound",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Learned fusion alpha")
    plt.legend()
    plt.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.45,
    )
    plt.tight_layout()

    fig.savefig(
        FUSION_ALPHA_FIGURE,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def append_batch_loss_records(records):
    """
    Append one completed epoch of train-batch losses to CSV.

    We intentionally write after an epoch finishes rather than every batch:
    - negligible I/O overhead;
    - resume is clean because checkpoints are also epoch-based;
    - an interrupted epoch leaves no duplicated partial rows on resume.
    """
    if not records:
        return

    df = pd.DataFrame(
        records
    )

    write_header = (
        not BATCH_LOSS_CSV.exists()
        or BATCH_LOSS_CSV.stat().st_size == 0
    )

    df.to_csv(
        BATCH_LOSS_CSV,
        mode="a",
        header=write_header,
        index=False,
    )


def append_gradient_diagnostic_records(records):
    """Append sampled train-batch gradient diagnostics to CSV."""
    if not records:
        return

    df = pd.DataFrame(records)
    write_header = (
        not GRAD_DIAGNOSTICS_CSV.exists()
        or GRAD_DIAGNOSTICS_CSV.stat().st_size == 0
    )
    df.to_csv(
        GRAD_DIAGNOSTICS_CSV,
        mode="a",
        header=write_header,
        index=False,
    )


def save_gradient_diagnostics_epoch_summary():
    """Create epoch summaries for backbone / physics / shared gradients."""
    if not GRAD_DIAGNOSTICS_CSV.exists():
        return
    df = pd.read_csv(GRAD_DIAGNOSTICS_CSV)
    if df.empty:
        return

    agg_kwargs = {"samples": ("batch", "count")}
    for scope in GRAD_DIAGNOSTIC_SCOPES:
        agg_kwargs.update({
            f"{scope}_grad_cls_norm_mean": (f"{scope}_grad_cls_norm", "mean"),
            f"{scope}_grad_other_norm_mean": (f"{scope}_grad_other_norm", "mean"),
            f"{scope}_grad_norm_ratio_mean": (
                f"{scope}_grad_norm_ratio_other_over_cls", "mean"
            ),
            f"{scope}_grad_cosine_mean": (f"{scope}_grad_cosine", "mean"),
            f"{scope}_grad_cosine_median": (f"{scope}_grad_cosine", "median"),
            f"{scope}_negative_cosine_rate": (
                f"{scope}_gradient_conflict", "mean"
            ),
        })

    summary = df.groupby("epoch", as_index=False).agg(**agg_kwargs)
    if "shared" in GRAD_DIAGNOSTIC_SCOPES:
        summary["grad_cls_norm_mean"] = summary["shared_grad_cls_norm_mean"]
        summary["grad_other_norm_mean"] = summary["shared_grad_other_norm_mean"]
        summary["grad_norm_ratio_mean"] = summary["shared_grad_norm_ratio_mean"]
        summary["grad_cosine_mean"] = summary["shared_grad_cosine_mean"]
        summary["grad_cosine_median"] = summary["shared_grad_cosine_median"]
        summary["negative_cosine_rate"] = summary["shared_negative_cosine_rate"]
    summary.to_csv(GRAD_DIAGNOSTICS_EPOCH_CSV, index=False)


def _mean_gradient_record_field(records, field):
    if not records:
        return float("nan")
    values = np.asarray([
        record.get(field, float("nan"))
        for record in records
    ], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def initialize_diagnostic_files():
    """
    Fresh run: remove old batch-loss CSV so results are never mixed.
    Resume run: keep it and append subsequent completed epochs.
    """
    if not RESUME:
        if BATCH_LOSS_CSV.exists():
            BATCH_LOSS_CSV.unlink()

        if FUSION_ALPHA_HISTORY_CSV.exists():
            FUSION_ALPHA_HISTORY_CSV.unlink()

        if FUSION_ALPHA_FIGURE.exists():
            FUSION_ALPHA_FIGURE.unlink()

        if GRAD_DIAGNOSTICS_CSV.exists():
            GRAD_DIAGNOSTICS_CSV.unlink()

        if GRAD_DIAGNOSTICS_EPOCH_CSV.exists():
            GRAD_DIAGNOSTICS_EPOCH_CSV.unlink()


def log_physics_shape_diagnostic(
    model,
    loader,
    logger,
):
    """
    One no-grad batch check before optimization.
    This catches profile/fusion shape mistakes early in the smoke test.
    """
    batch = next(
        iter(loader)
    )

    images = batch[
        "image"
    ].to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            images
        )

    logger.info(
        "Shape check    : image=%s",
        tuple(
            images.shape
        ),
    )

    if outputs[
        "physics_profile"
    ] is not None:
        logger.info(
            "Physics prof. : %s",
            tuple(
                outputs[
                    "physics_profile"
                ].shape
            ),
        )

    if outputs[
        "physics_feature"
    ] is not None:
        logger.info(
            "Physics feat. : %s",
            tuple(
                outputs[
                    "physics_feature"
                ].shape
            ),
        )

    if outputs.get("physics_feature_cls") is not None:
        logger.info(
            "Physics cls   : %s",
            tuple(outputs["physics_feature_cls"].shape),
        )

    if outputs.get("physics_feature_loc") is not None:
        logger.info(
            "Physics loc   : %s",
            tuple(outputs["physics_feature_loc"].shape),
        )

    logger.info(
        "Cls logits     : %s",
        tuple(
            outputs[
                "classification_logits"
            ].shape
        ),
    )

    logger.info(
        "Heatmap logits : %s",
        tuple(
            outputs[
                "heatmap_logits"
            ].shape
        ),
    )

    logger.info(
        "Offset logits  : %s",
        tuple(
            outputs[
                "offset_logits"
            ].shape
        ),
    )

    model.train()


# =============================================================================
# 4. Data / normalization
# =============================================================================

def compute_train_normalization(logger):
    logger.info(
        "Computing normalization from TRAIN split only: %s",
        NORMALIZATION_SOURCE_CSV,
    )

    raw_train = DarkSolitonDataset(
        csv_path=NORMALIZATION_SOURCE_CSV,
        task="classification",
        normalize="none",
        check_file_exists=CHECK_FILE_EXISTS,
    )

    mean, std = compute_dataset_mean_std(
        raw_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    if std <= 0:
        raise RuntimeError(f"Invalid train std={std}")

    payload = {
        "method": "dataset_zscore",
        "mean": float(mean),
        "std": float(std),
        "source": str(NORMALIZATION_SOURCE_CSV),
        "note": "Computed from train.csv only; validation/test were not used.",
    }

    with open(METRICS_DIR / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Normalization: mean=%.8f, std=%.8f", mean, std)
    return float(mean), float(std)


def build_datasets_and_loaders(mean, std):
    common = dict(
        task="classification",
        normalize="dataset_zscore",
        dataset_mean=mean,
        dataset_std=std,
        check_file_exists=CHECK_FILE_EXISTS,
    )

    train_dataset = DarkSolitonDataset(csv_path=TRAIN_CSV, **common)
    val_dataset = DarkSolitonDataset(csv_path=VAL_CSV, **common)
    test_dataset = DarkSolitonDataset(csv_path=TEST_CSV, **common)

    pin = DEVICE.type == "cuda"

    train_loader = make_dataloader(
        train_dataset, BATCH_SIZE, True, NUM_WORKERS, pin
    )
    val_loader = make_dataloader(
        val_dataset, BATCH_SIZE, False, NUM_WORKERS, pin
    )
    test_loader = make_dataloader(
        test_dataset, BATCH_SIZE, False, NUM_WORKERS, pin
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def compute_class_weights(train_dataset):
    label_col = (
        "training_label"
        if "training_label" in train_dataset.df.columns
        else "label_v3"
    )

    labels = pd.to_numeric(
        train_dataset.df[label_col], errors="raise"
    ).astype(int)

    counts = labels.value_counts().sort_index()
    expected = set(range(NUM_CLASSES))

    if set(counts.index.tolist()) != expected:
        raise ValueError(
            f"Training labels must contain {sorted(expected)}, "
            f"got {counts.to_dict()}"
        )

    total = len(labels)
    weights = [
        total / (NUM_CLASSES * int(counts.loc[c]))
        for c in range(NUM_CLASSES)
    ]

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# =============================================================================
# 5. Multi-task localization targets
# =============================================================================

def build_multitask_localization_targets(batch):
    """
    class-0: localization-valid, all-zero heatmap
    class-1: localization-valid, Gaussian heatmap + offset
    class-2: localization-invalid, completely masked from localization loss
    """
    labels = batch["label"].to(
        DEVICE, non_blocking=(DEVICE.type == "cuda")
    ).long()

    positions = batch["position"].to(
        DEVICE, non_blocking=(DEVICE.type == "cuda")
    ).float()

    position_valid = batch["position_valid"].to(
        DEVICE, non_blocking=(DEVICE.type == "cuda")
    ).bool()

    batch_size = labels.shape[0]
    localization_valid = (labels == 0) | (labels == 1)
    positive_mask = labels == 1

    invalid_positive = positive_mask & (~position_valid)
    if invalid_positive.any():
        bad = torch.nonzero(invalid_positive, as_tuple=False).squeeze(1)
        raise ValueError(
            "class-1 sample missing valid excitation position; "
            f"batch indices={bad.tolist()}"
        )

    heatmap_target = torch.zeros(
        batch_size, NUM_BINS, dtype=torch.float32, device=DEVICE
    )
    offset_target = torch.zeros_like(heatmap_target)
    offset_mask = torch.zeros(
        batch_size, NUM_BINS, dtype=torch.bool, device=DEVICE
    )

    positive_rows = torch.nonzero(positive_mask, as_tuple=False).squeeze(1)

    if positive_rows.numel() > 0:
        positive_positions = positions[positive_mask]

        if (
            (positive_positions < 0.0).any()
            or (positive_positions > (IMAGE_WIDTH - 1)).any()
        ):
            raise ValueError("class-1 GT position outside [0, 163].")

        scaled = positive_positions / BIN_WIDTH
        center_bins = torch.floor(scaled).long().clamp(0, NUM_BINS - 1)
        offsets = (scaled - center_bins.float()).clamp(0.0, 1.0 - 1e-7)

        for row_idx, center_bin, offset in zip(
            positive_rows.tolist(),
            center_bins.tolist(),
            offsets.tolist(),
        ):
            left = max(0, center_bin - GAUSSIAN_RADIUS_BINS)
            right = min(NUM_BINS - 1, center_bin + GAUSSIAN_RADIUS_BINS)

            bins = torch.arange(
                left, right + 1, device=DEVICE, dtype=torch.float32
            )
            d = bins - float(center_bin)
            gaussian = torch.exp(
                -0.5 * (d / GAUSSIAN_SIGMA_BINS) ** 2
            )

            heatmap_target[row_idx, left:right + 1] = torch.maximum(
                heatmap_target[row_idx, left:right + 1], gaussian
            )
            heatmap_target[row_idx, center_bin] = 1.0

            offset_target[row_idx, center_bin] = float(offset)
            offset_mask[row_idx, center_bin] = True

    return {
        "labels": labels,
        "localization_valid": localization_valid,
        "heatmap_target": heatmap_target,
        "offset_target": offset_target,
        "offset_mask": offset_mask,
    }


# =============================================================================
# 6. Multi-task loss
# =============================================================================

class PerImageNormalizedModifiedFocalLoss(nn.Module):
    """
    CenterNet-style focal loss with stable per-image reduction.

    For a class-1 image (one positive center), the full focal loss is normalized
    by the number of positive centers. For a class-0 image (no center), the
    negative-only loss is averaged across bins instead of being summed. The final
    loss is the mean over images, preventing random class-0/class-1 batch
    composition from directly changing the loss scale.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, logits, target):
        if logits.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "PerImageNormalizedModifiedFocalLoss expects [B, num_bins]."
            )

        pred = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        positive_mask = (target >= 1.0 - 1e-6).float()
        negative_mask = (target < 1.0 - 1e-6).float()
        negative_weights = (1.0 - target) ** self.beta

        positive_loss = (
            -torch.log(pred)
            * ((1.0 - pred) ** self.alpha)
            * positive_mask
        )
        negative_loss = (
            -torch.log(1.0 - pred)
            * (pred ** self.alpha)
            * negative_weights
            * negative_mask
        )

        pos_sum = positive_loss.sum(dim=1)
        neg_sum = negative_loss.sum(dim=1)
        num_positive = positive_mask.sum(dim=1)
        num_bins = float(logits.shape[1])

        positive_rows = num_positive > 0
        per_image = torch.where(
            positive_rows,
            (pos_sum + neg_sum) / num_positive.clamp_min(1.0),
            neg_sum / max(num_bins, 1.0),
        )
        return per_image.mean()


def _task_losses_for_gradient_control(losses):
    """Return weighted task-level losses whose sum equals the optimized loss."""
    task_cls = (
        LAMBDA_CLASSIFICATION
        * losses["loss_classification"]
    )
    task_other = (
        LAMBDA_LOCALIZATION
        * losses["loss_localization"]
        + LAMBDA_PHYSICS
        * losses["loss_physics"]
    )
    return task_cls, task_other


def _gradient_diagnostic_parameters(model, scope):
    """Return unique trainable parameters for one diagnostic scope.

    V2.2 adds physics_cls / physics_loc to verify that the two independent
    physics paths are truly task-specific. In dual_task mode each should have
    gradients from only one task and therefore no cross-task cosine.
    """
    scope = str(scope)

    cls_modules = []
    if getattr(model, "physics_encoder_cls", None) is not None:
        cls_modules.append(model.physics_encoder_cls)
    if getattr(model, "physics_cls_fusion", None) is not None:
        cls_modules.append(model.physics_cls_fusion)

    loc_modules = []
    if getattr(model, "physics_encoder", None) is not None:
        loc_modules.append(model.physics_encoder)
    if getattr(model, "physics_fusions", None) is not None:
        loc_modules.append(model.physics_fusions)
    if getattr(model, "physics_aux_head", None) is not None:
        loc_modules.append(model.physics_aux_head)

    if scope == "backbone":
        modules = [model.backbone]
    elif scope == "physics_cls":
        modules = cls_modules
    elif scope == "physics_loc":
        modules = loc_modules
    elif scope == "physics":
        modules = cls_modules + loc_modules
    elif scope == "shared":
        modules = [model.backbone] + cls_modules + loc_modules
    else:
        raise ValueError(f"Unsupported gradient diagnostic scope={scope!r}")

    params = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    return params


def _gradient_stats_from_lists(grads_a, grads_b, eps=1e-12):
    """
    Gradient statistics that remain meaningful for task-specific modules.

    In V2.2 dual_task routing, physics_cls parameters receive classification
    gradients only, while physics_loc parameters receive localization gradients
    only. Each task norm therefore uses all available gradients, while cosine/
    dot/conflict are computed only on parameter tensors used by BOTH tasks. If
    a scope is task-specific and has no common tensors, cosine/conflict are NaN.
    """
    dot = None
    norm_a_sq = None
    norm_b_sq = None
    common_norm_a_sq = None
    common_norm_b_sq = None
    common_tensors = 0
    cls_only_tensors = 0
    other_only_tensors = 0

    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is not None:
            a2 = torch.sum(grad_a * grad_a)
            norm_a_sq = a2 if norm_a_sq is None else norm_a_sq + a2
        if grad_b is not None:
            b2 = torch.sum(grad_b * grad_b)
            norm_b_sq = b2 if norm_b_sq is None else norm_b_sq + b2

        if grad_a is None and grad_b is None:
            continue
        if grad_a is None:
            other_only_tensors += 1
            continue
        if grad_b is None:
            cls_only_tensors += 1
            continue

        common_tensors += 1
        d = torch.sum(grad_a * grad_b)
        a2_common = torch.sum(grad_a * grad_a)
        b2_common = torch.sum(grad_b * grad_b)
        dot = d if dot is None else dot + d
        common_norm_a_sq = (
            a2_common
            if common_norm_a_sq is None
            else common_norm_a_sq + a2_common
        )
        common_norm_b_sq = (
            b2_common
            if common_norm_b_sq is None
            else common_norm_b_sq + b2_common
        )

    device = None
    for g in list(grads_a) + list(grads_b):
        if g is not None:
            device = g.device
            break
    zero = torch.tensor(0.0, device=device)

    norm_a = torch.sqrt(
        (norm_a_sq if norm_a_sq is not None else zero).clamp_min(0.0)
    )
    norm_b = torch.sqrt(
        (norm_b_sq if norm_b_sq is not None else zero).clamp_min(0.0)
    )

    norm_a_value = float(norm_a.detach().cpu().item())
    norm_b_value = float(norm_b.detach().cpu().item())
    ratio_value = (
        norm_b_value / (norm_a_value + float(eps))
        if norm_a_value > float(eps)
        else float("nan")
    )

    if common_tensors == 0:
        dot_value = float("nan")
        cosine_value = float("nan")
        conflict_value = float("nan")
    else:
        common_norm_a = torch.sqrt(common_norm_a_sq.clamp_min(0.0))
        common_norm_b = torch.sqrt(common_norm_b_sq.clamp_min(0.0))
        cosine = dot / (common_norm_a * common_norm_b + float(eps))
        dot_value = float(dot.detach().cpu().item())
        cosine_value = float(cosine.detach().cpu().item())
        conflict_value = float((dot < 0).detach().cpu().item())

    return {
        "grad_cls_norm": norm_a_value,
        "grad_other_norm": norm_b_value,
        "grad_norm_ratio_other_over_cls": ratio_value,
        "grad_dot": dot_value,
        "grad_cosine": cosine_value,
        "gradient_conflict": conflict_value,
        "common_gradient_tensors": int(common_tensors),
        "cls_only_gradient_tensors": int(cls_only_tensors),
        "other_only_gradient_tensors": int(other_only_tensors),
    }


def compute_gradient_diagnostics(model, task_cls_loss, task_other_loss):
    """Compute backbone / task-specific physics / combined task-gradient diagnostics."""
    scope_params = {
        scope: _gradient_diagnostic_parameters(model, scope)
        for scope in GRAD_DIAGNOSTIC_SCOPES
    }

    union_params = []
    seen = set()
    for params in scope_params.values():
        for parameter in params:
            if id(parameter) not in seen:
                union_params.append(parameter)
                seen.add(id(parameter))

    if not union_params:
        result = {}
        empty = _gradient_stats_from_lists([], [], GRAD_DIAGNOSTIC_EPS)
        for scope in GRAD_DIAGNOSTIC_SCOPES:
            for key, value in empty.items():
                result[f"{scope}_{key}"] = value
        return result

    # Only two autograd passes over the UNION; individual scopes are then
    # reconstructed by parameter identity. This is much cheaper than three
    # independent pairs of autograd.grad calls.
    grads_cls = torch.autograd.grad(
        task_cls_loss, union_params, retain_graph=True, allow_unused=True
    )
    grads_other = torch.autograd.grad(
        task_other_loss, union_params, retain_graph=True, allow_unused=True
    )
    cls_by_id = {id(p): g for p, g in zip(union_params, grads_cls)}
    other_by_id = {id(p): g for p, g in zip(union_params, grads_other)}

    result = {}
    for scope, params in scope_params.items():
        stats = _gradient_stats_from_lists(
            [cls_by_id.get(id(p)) for p in params],
            [other_by_id.get(id(p)) for p in params],
            GRAD_DIAGNOSTIC_EPS,
        )
        for key, value in stats.items():
            result[f"{scope}_{key}"] = value

    # Legacy aliases represent the complete shared path.
    for key in [
        "grad_cls_norm", "grad_other_norm",
        "grad_norm_ratio_other_over_cls", "grad_dot",
        "grad_cosine", "gradient_conflict", "common_gradient_tensors",
    ]:
        result[key] = result.get(f"shared_{key}", float("nan"))
    return result


def apply_pcgrad(
    model,
    task_cls_loss,
    task_other_loss,
):
    """
    Deterministic two-task PCGrad.

    Gradients are computed separately. If the two task gradients conflict on a
    parameter tensor (negative dot product), each gradient is projected against
    the other using the original gradients, then the two projected gradients are
    summed. Parameters used by only one task keep that task gradient unchanged.
    """
    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    grads_cls = torch.autograd.grad(
        task_cls_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    grads_other = torch.autograd.grad(
        task_other_loss,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    diagnostic_ids = {
        id(parameter)
        for parameter in _gradient_diagnostic_parameters(model, "shared")
    }
    diag_cls = []
    diag_other = []

    projected_conflicts = 0
    common_tensors = 0

    for parameter, grad_cls, grad_other in zip(
        params,
        grads_cls,
        grads_other,
    ):
        if id(parameter) in diagnostic_ids:
            diag_cls.append(grad_cls)
            diag_other.append(grad_other)

        if grad_cls is None and grad_other is None:
            parameter.grad = None
            continue
        if grad_cls is None:
            parameter.grad = grad_other.detach().clone()
            continue
        if grad_other is None:
            parameter.grad = grad_cls.detach().clone()
            continue

        common_tensors += 1
        dot = torch.sum(grad_cls * grad_other)

        if dot < 0:
            projected_conflicts += 1
            cls_norm_sq = torch.sum(grad_cls * grad_cls)
            other_norm_sq = torch.sum(grad_other * grad_other)

            projected_cls = (
                grad_cls
                - dot / (other_norm_sq + PCGRAD_EPS)
                * grad_other
            )
            projected_other = (
                grad_other
                - dot / (cls_norm_sq + PCGRAD_EPS)
                * grad_cls
            )
            combined = projected_cls + projected_other
        else:
            combined = grad_cls + grad_other

        parameter.grad = combined.detach().clone()

    stats = _gradient_stats_from_lists(
        diag_cls,
        diag_other,
        GRAD_DIAGNOSTIC_EPS,
    )
    stats["pcgrad_projected_conflict_tensors"] = int(projected_conflicts)
    stats["pcgrad_common_gradient_tensors"] = int(common_tensors)
    return stats


class ModelEMA:
    """Simple parameter/buffer EMA used only when USE_EMA=True."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        source_state = model.state_dict()
        ema_state = self.model.state_dict()

        use_decay = (
            self.decay
            if self.num_updates > int(EMA_UPDATE_AFTER_STEP)
            else 0.0
        )

        for key, ema_value in ema_state.items():
            source_value = source_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(source_value)
            else:
                ema_value.mul_(use_decay).add_(
                    source_value,
                    alpha=1.0 - use_decay,
                )

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "model_state_dict": self.model.state_dict(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        self.model.load_state_dict(state["model_state_dict"])


def compute_multitask_loss(
    outputs,
    batch,
    classification_criterion,
    heatmap_criterion,
    offset_criterion,
):
    targets = build_multitask_localization_targets(
        batch
    )
    labels = targets["labels"]

    # Classification: all class-0/1/2
    loss_cls = classification_criterion(
        outputs["classification_logits"],
        labels,
    )

    # Main heatmap: class-0/1 only; class-2 is NOT background.
    loc_valid = targets[
        "localization_valid"
    ]

    if loc_valid.any():
        loss_heatmap = heatmap_criterion(
            outputs[
                "heatmap_logits"
            ][loc_valid],
            targets[
                "heatmap_target"
            ][loc_valid],
        )
    else:
        loss_heatmap = (
            outputs[
                "heatmap_logits"
            ].sum()
            * 0.0
        )

    # Offset: class-1 GT center bin only.
    offset_mask = targets[
        "offset_mask"
    ]

    if offset_mask.any():
        pred_offsets = torch.sigmoid(
            outputs[
                "offset_logits"
            ][offset_mask]
        )

        gt_offsets = targets[
            "offset_target"
        ][offset_mask]

        loss_offset = offset_criterion(
            pred_offsets,
            gt_offsets,
        )
    else:
        loss_offset = (
            outputs[
                "offset_logits"
            ].sum()
            * 0.0
        )

    loss_loc = (
        LAMBDA_HEATMAP
        * loss_heatmap
        + LAMBDA_OFFSET
        * loss_offset
    )

    # Optional physics-only heatmap auxiliary loss.
    # Main V1: disabled, so this is exactly zero.
    aux_logits = outputs.get(
        "physics_aux_heatmap_logits"
    )

    if (
        USE_PHYSICS_AUX_LOSS
        and aux_logits is not None
        and loc_valid.any()
    ):
        loss_physics = heatmap_criterion(
            aux_logits[
                loc_valid
            ],
            targets[
                "heatmap_target"
            ][loc_valid],
        )
    elif aux_logits is not None:
        loss_physics = (
            aux_logits.sum()
            * 0.0
        )
    else:
        loss_physics = (
            outputs[
                "classification_logits"
            ].sum()
            * 0.0
        )

    loss_total = (
        LAMBDA_CLASSIFICATION
        * loss_cls
        + LAMBDA_LOCALIZATION
        * loss_loc
        + LAMBDA_PHYSICS
        * loss_physics
    )

    return {
        "loss": loss_total,
        "loss_classification": (
            loss_cls
        ),
        "loss_localization": (
            loss_loc
        ),
        "loss_heatmap": (
            loss_heatmap
        ),
        "loss_offset": (
            loss_offset
        ),
        "loss_physics": (
            loss_physics
        ),
    }


# =============================================================================
# 7. Metrics / decoding
# =============================================================================

def compute_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def decode_localization_cache(localization_cache, threshold):
    """Decode cached class-0/1 outputs without another network forward pass."""
    records = []

    for item in localization_cache:
        heatmap_probs = item["heatmap_probabilities"]
        offset_probs = item["offset_probabilities"]

        peak_bins = detector_base.find_local_peak_bins(
            probabilities=heatmap_probs,
            threshold=float(threshold),
        )

        pred_positions = [
            detector_base.continuous_position_from_bin(
                bin_index=b,
                offset_probability=offset_probs[b],
            )
            for b in peak_bins
        ]
        pred_scores = [float(heatmap_probs[b]) for b in peak_bins]

        top1_bin = int(np.argmax(heatmap_probs))
        top1_position = detector_base.continuous_position_from_bin(
            bin_index=top1_bin,
            offset_probability=offset_probs[top1_bin],
        )

        records.append({
            "file_name": item["file_name"],
            "label": item["label"],
            "gt_position": item["gt_position"],
            "pred_positions": pred_positions,
            "pred_scores": pred_scores,
            "top1_position": top1_position,
            "top1_score": float(heatmap_probs[top1_bin]),
        })

    return records


def localization_metrics_from_cache(localization_cache, threshold):
    records = decode_localization_cache(localization_cache, threshold)
    metrics = detector_base.evaluate_prediction_records(
        records,
        primary_tolerance_px=PRIMARY_TOLERANCE_PX,
    )
    return metrics, records


# =============================================================================
# 8. Train / evaluate
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    amp_enabled,
    classification_criterion,
    heatmap_criterion,
    offset_criterion,
    epoch,
    ema=None,
):
    model.train()

    keys = [
        "loss",
        "loss_classification",
        "loss_localization",
        "loss_heatmap",
        "loss_offset",
        "loss_physics",
    ]
    sums = {k: 0.0 for k in keys}
    total_samples = 0
    batch_records = []
    gradient_records = []
    samples_seen_epoch = 0

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(
            DEVICE, non_blocking=(DEVICE.type == "cuda")
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)
            losses = compute_multitask_loss(
                outputs,
                batch,
                classification_criterion,
                heatmap_criterion,
                offset_criterion,
            )
            task_cls_loss, task_other_loss = (
                _task_losses_for_gradient_control(
                    losses
                )
            )

        should_record_grad = (
            USE_GRADIENT_DIAGNOSTICS
            and (
                batch_index == 1
                or batch_index % int(GRAD_DIAGNOSTIC_INTERVAL) == 0
            )
        )

        grad_stats = None

        # Record NATIVE task interaction before any PCGrad projection.
        if should_record_grad:
            grad_stats = compute_gradient_diagnostics(
                model, task_cls_loss, task_other_loss
            )

        if USE_PCGRAD:
            # PCGrad uses unscaled autograd.grad; AMP is disabled in main().
            pcgrad_stats = apply_pcgrad(
                model, task_cls_loss, task_other_loss
            )
            if grad_stats is not None:
                for key, value in pcgrad_stats.items():
                    if key.startswith("pcgrad_"):
                        grad_stats[key] = value

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )
            optimizer.step()

        else:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

        if ema is not None:
            ema.update(model)

        bs = int(images.shape[0])

        for k in keys:
            sums[k] += losses[k].item() * bs

        total_samples += bs
        samples_seen_epoch += bs

        lr_values = optimizer_lr_dict(optimizer)
        current_lr = float(optimizer.param_groups[0]["lr"])

        batch_records.append({
            "epoch": int(epoch),
            "batch": int(batch_index),
            "global_step": int(
                (epoch - 1) * len(loader) + batch_index
            ),
            "batch_size": bs,
            "samples_seen_epoch": int(samples_seen_epoch),
            "lr": current_lr,
            **lr_values,
            "loss": float(losses["loss"].detach().item()),
            "loss_classification": float(
                losses["loss_classification"].detach().item()
            ),
            "loss_localization": float(
                losses["loss_localization"].detach().item()
            ),
            "loss_heatmap": float(
                losses["loss_heatmap"].detach().item()
            ),
            "loss_offset": float(
                losses["loss_offset"].detach().item()
            ),
            "loss_physics": float(
                losses["loss_physics"].detach().item()
            ),
        })

        if grad_stats is not None:
            gradient_records.append({
                "epoch": int(epoch),
                "batch": int(batch_index),
                "global_step": int(
                    (epoch - 1) * len(loader) + batch_index
                ),
                "diagnostic_scopes": ",".join(GRAD_DIAGNOSTIC_SCOPES),
                "pcgrad_enabled": bool(USE_PCGRAD),
                **grad_stats,
            })

    result = {
        k: v / total_samples
        for k, v in sums.items()
    }
    result["batch_records"] = batch_records
    result["gradient_records"] = gradient_records
    return result


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    classification_criterion,
    heatmap_criterion,
    offset_criterion,
    localization_threshold,
    collect_features=False,
):
    model.eval()

    keys = [
        "loss",
        "loss_classification",
        "loss_localization",
        "loss_heatmap",
        "loss_offset",
        "loss_physics",
    ]
    sums = {k: 0.0 for k in keys}
    total_samples = 0

    y_true, y_pred = [], []
    probability_rows, file_names = [], []
    feature_rows = []
    localization_cache_01 = []
    localization_cache_c2 = []

    dataset_df = getattr(loader.dataset, "df", None)
    has_manual_validation = (
        dataset_df is not None
        and VAL_MANUAL_STATUS_COLUMN in dataset_df.columns
        and VAL_MANUAL_POSITIONS_COLUMN in dataset_df.columns
        and VAL_LOC_INCLUDE_COLUMN in dataset_df.columns
    )

    for batch in loader:
        images = batch["image"].to(
            DEVICE, non_blocking=(DEVICE.type == "cuda")
        )

        outputs = model(images)
        losses = compute_multitask_loss(
            outputs,
            batch,
            classification_criterion,
            heatmap_criterion,
            offset_criterion,
        )

        bs = images.shape[0]
        for k in keys:
            sums[k] += losses[k].item() * bs
        total_samples += bs

        labels = batch["label"].cpu().numpy().astype(int)
        sample_indices = batch["index"].cpu().numpy().astype(int)

        probs = torch.softmax(
            outputs["classification_logits"], dim=1
        ).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        probability_rows.extend(probs.tolist())
        file_names.extend(list(batch["file_name"]))

        if collect_features:
            feature_rows.extend(
                outputs["shared_feature"].cpu().numpy().tolist()
            )

        hm_probs = torch.sigmoid(
            outputs["heatmap_logits"]
        ).cpu().numpy()
        off_probs = torch.sigmoid(
            outputs["offset_logits"]
        ).cpu().numpy()

        positions = batch["position"].cpu().numpy().astype(float)

        for i in range(bs):
            label = int(labels[i])
            common = {
                "file_name": batch["file_name"][i],
                "label": label,
                "heatmap_probabilities": hm_probs[i].astype(np.float32),
                "offset_probabilities": off_probs[i].astype(np.float32),
            }

            # Original strict class-0/1 validation is preserved exactly.
            if label in {0, 1}:
                localization_cache_01.append({
                    **common,
                    "gt_position": float(positions[i]) if label == 1 else None,
                })

            # V2.3-VAL: manually annotated class-2 multi-target validation.
            if label == 2 and has_manual_validation:
                row = dataset_df.iloc[int(sample_indices[i])]
                include = _safe_bool(row.get(VAL_LOC_INCLUDE_COLUMN, False))
                status = str(
                    row.get(VAL_MANUAL_STATUS_COLUMN, "")
                ).strip().lower()

                if include:
                    if status not in VAL_LOC_ELIGIBLE_CLASS2_STATUSES:
                        raise ValueError(
                            f"{row['file_name']}: loc_eval_include=True but status={status!r}"
                        )
                    gt_positions = parse_position_list(
                        row.get(VAL_MANUAL_POSITIONS_COLUMN, "")
                    )
                    if status == "valid" and len(gt_positions) == 0:
                        raise ValueError(
                            f"{row['file_name']}: valid class-2 sample has no manual GT."
                        )
                    if status == "no_reliable_soliton" and gt_positions:
                        raise ValueError(
                            f"{row['file_name']}: no_reliable_soliton must have [] GT."
                        )
                    localization_cache_c2.append({
                        **common,
                        "gt_positions": gt_positions,
                        "manual_status": status,
                    })

    cls_metrics = compute_classification_metrics(
        np.asarray(y_true), np.asarray(y_pred)
    )

    loc01_metrics, loc01_records = localization_metrics_from_cache(
        localization_cache_01, localization_threshold
    )
    # Add exact record count for combined micro diagnostics.
    loc01_metrics = dict(loc01_metrics)
    loc01_metrics["record_count"] = int(len(localization_cache_01))
    loc01_metrics["num_images"] = int(len(localization_cache_01))

    if localization_cache_c2:
        loc_c2_metrics, loc_c2_records = multitarget_localization_metrics_from_cache(
            localization_cache_c2, localization_threshold
        )
    else:
        loc_c2_metrics, loc_c2_records = None, []

    loc_summary = combine_localization_scores(loc01_metrics, loc_c2_metrics)
    loc_score = float(loc_summary["loc_score"])
    joint_score = harmonic_joint_score(
        cls_metrics["macro_f1"], loc_score
    )

    return {
        **{k: v / total_samples for k, v in sums.items()},
        "classification_metrics": cls_metrics,
        # Backward-compatible aliases: strict class-0/1 localization.
        "localization_metrics": loc01_metrics,
        "localization_metrics_01": loc01_metrics,
        "localization_metrics_c2": loc_c2_metrics,
        "localization_score": loc_score,
        "localization_summary": loc_summary,
        "joint_score": joint_score,
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "probabilities": np.asarray(probability_rows, dtype=np.float32),
        "file_names": file_names,
        "localization_cache": localization_cache_01,
        "localization_cache_01": localization_cache_01,
        "localization_cache_c2": localization_cache_c2,
        "localization_records": loc01_records,
        "localization_records_01": loc01_records,
        "localization_records_c2": loc_c2_records,
        "features": (
            np.asarray(feature_rows, dtype=np.float32)
            if collect_features
            else None
        ),
    }


# =============================================================================
# 9. Validation threshold search — cached, no repeated inference
# =============================================================================

def search_best_localization_threshold(
    localization_cache_01,
    localization_cache_c2,
):
    """
    Validation-only threshold calibration for V2.3-VAL.

    Primary objective:
        LocScore = H(F1_class0/1, F1_class2_manual)

    Tie-breaks:
        1) combined micro precision higher
        2) combined FP/image lower
        3) threshold higher

    Legacy columns (f1_at_2px / precision_at_2px / false_positives_per_image)
    are retained so existing Stage-4 Gold evaluators can read the selected
    threshold. Here f1_at_2px intentionally aliases LocScore.
    """
    if not localization_cache_c2 and REQUIRE_CLASS2_VAL_ANNOTATIONS:
        raise RuntimeError(
            "Threshold search requires eligible class-2 validation annotations."
        )

    rows = []
    best_threshold = None
    best_bundle = None
    best_score = -1.0

    for threshold in THRESHOLD_SEARCH_VALUES:
        loc01, _ = localization_metrics_from_cache(
            localization_cache_01, float(threshold)
        )
        loc01 = dict(loc01)
        loc01["record_count"] = int(len(localization_cache_01))
        loc01["num_images"] = int(len(localization_cache_01))

        loc_c2, _ = multitarget_localization_metrics_from_cache(
            localization_cache_c2, float(threshold)
        )

        summary = combine_localization_scores(loc01, loc_c2)
        loc_score = float(summary["loc_score"])
        combined_precision = float(summary["combined_precision_at_2px"])
        combined_recall = float(summary["combined_recall_at_2px"])
        combined_micro_f1 = float(summary["combined_micro_f1_at_2px"])
        combined_fp_per_image = float(summary["combined_false_positives_per_image"])

        # Harmonized accuracy-like diagnostic across the two localization tasks.
        acc1_c2 = loc_c2["gt_match_rate_at_1px"] if loc_c2 is not None else loc01["acc_at_1px"]
        acc2_c2 = loc_c2["gt_match_rate_at_2px"] if loc_c2 is not None else loc01["acc_at_2px"]
        acc3_c2 = loc_c2["gt_match_rate_at_3px"] if loc_c2 is not None else loc01["acc_at_3px"]

        row = {
            "threshold": float(threshold),
            # Backward-compatible aliases used by evaluator threshold loading.
            "precision_at_2px": combined_precision,
            "recall_at_2px": combined_recall,
            "f1_at_2px": loc_score,
            "acc_at_1px": harmonic_joint_score(loc01["acc_at_1px"], acc1_c2),
            "acc_at_2px": harmonic_joint_score(loc01["acc_at_2px"], acc2_c2),
            "acc_at_3px": harmonic_joint_score(loc01["acc_at_3px"], acc3_c2),
            "top1_mae_px": loc01["top1_mae_px"],
            "matched_mae_px": summary["combined_matched_mae_px"],
            "false_positives_per_image": combined_fp_per_image,

            "loc_score": loc_score,
            "combined_micro_precision_at_2px": combined_precision,
            "combined_micro_recall_at_2px": combined_recall,
            "combined_micro_f1_at_2px": combined_micro_f1,

            "loc01_precision_at_2px": loc01["precision_at_2px"],
            "loc01_recall_at_2px": loc01["recall_at_2px"],
            "loc01_f1_at_2px": loc01["f1_at_2px"],
            "loc01_matched_mae_px": loc01["matched_mae_px"],
            "loc01_false_positives_per_image": loc01["false_positives_per_image"],

            "loc_c2_precision_at_2px": loc_c2["precision_at_2px"],
            "loc_c2_recall_at_2px": loc_c2["recall_at_2px"],
            "loc_c2_f1_at_2px": loc_c2["f1_at_2px"],
            "loc_c2_matched_mae_px": loc_c2["matched_mae_px"],
            "loc_c2_exact_count_accuracy": loc_c2["exact_count_accuracy"],
            "loc_c2_exact_sample_success_at_2px": loc_c2["exact_sample_success_at_2px"],
            "loc_c2_count_mae": loc_c2["count_mae"],
            "loc_c2_false_positives_per_image": loc_c2["false_positives_per_image"],
            "loc_c2_false_negatives_per_image": loc_c2["false_negatives_per_image"],
            "loc_c2_num_images": loc_c2["num_images"],
            "loc_c2_total_gt_excitations": loc_c2["total_gt_excitations"],
        }
        rows.append(row)

        better = loc_score > best_score + 1e-12
        if (
            not better
            and best_bundle is not None
            and abs(loc_score - best_score) <= 1e-12
        ):
            best_precision = float(
                best_bundle["summary"]["combined_precision_at_2px"]
            )
            if combined_precision > best_precision + 1e-12:
                better = True
            elif abs(combined_precision - best_precision) <= 1e-12:
                best_fp = float(
                    best_bundle["summary"]["combined_false_positives_per_image"]
                )
                if combined_fp_per_image < best_fp - 1e-12:
                    better = True
                elif (
                    abs(combined_fp_per_image - best_fp) <= 1e-12
                    and float(threshold) > float(best_threshold)
                ):
                    better = True

        if better:
            best_threshold = float(threshold)
            best_score = loc_score
            best_bundle = {
                "loc01": loc01,
                "loc_c2": loc_c2,
                "loc_score": loc_score,
                "summary": summary,
            }

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(
        METRICS_DIR / "val_threshold_search.csv",
        index=False,
    )

    return best_threshold, best_bundle, threshold_df


# =============================================================================
# 9B. Stage-4 V2 optimization helpers
# =============================================================================

def set_visual_backbone_trainable(model, trainable):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(bool(trainable))


def build_optimizer(model):
    if not USE_DIFFERENTIAL_LR:
        return AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    groups = {
        "backbone": {"params": [], "lr": LR_BACKBONE},
        "task_heads": {"params": [], "lr": LR_TASK_HEADS},
        "physics": {"params": [], "lr": LR_PHYSICS},
        "adapters": {"params": [], "lr": LR_ADAPTERS},
        "other": {"params": [], "lr": LEARNING_RATE},
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            # Keep frozen parameters out of AdamW state until they are unfrozen.
            # If warm-start freezing is requested, freezing is applied after
            # optimizer construction so all parameters remain represented.
            pass

        if name.startswith("backbone."):
            key = "backbone"
        elif name.startswith(("classification_head.", "localization_head.")):
            key = "task_heads"
        elif name.startswith((
            "physics_encoder.",
            "physics_fusions.",
            "physics_encoder_cls.",
            "physics_cls_fusion.",
            "physics_aux_head.",
        )):
            key = "physics"
        elif name.startswith((
            "classification_adapters.",
            "localization_adapters.",
        )):
            key = "adapters"
        else:
            key = "other"

        groups[key]["params"].append(parameter)

    param_groups = []
    for group_name, group in groups.items():
        if not group["params"]:
            continue
        param_groups.append({
            "params": group["params"],
            "lr": float(group["lr"]),
            "weight_decay": WEIGHT_DECAY,
            "name": group_name,
        })

    return AdamW(param_groups)


def optimizer_lr_dict(optimizer):
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group{index}"))
        result[f"lr_{name}"] = float(group["lr"])
    return result


def maybe_update_backbone_freeze_state(model, epoch, logger):
    if INIT_MODE != "stage3":
        return

    freeze_epochs = int(WARM_START_FREEZE_BACKBONE_EPOCHS)
    should_train = epoch > freeze_epochs

    current = any(
        parameter.requires_grad
        for parameter in model.backbone.parameters()
    )

    if current != should_train:
        set_visual_backbone_trainable(
            model,
            should_train,
        )
        logger.info(
            "Visual backbone trainable=%s at epoch %d "
            "(warm-start freeze epochs=%d)",
            should_train,
            epoch,
            freeze_epochs,
        )


# =============================================================================
# 10. Checkpoints
# =============================================================================

def save_checkpoint(
    path_,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_joint_score,
    best_epoch,
    no_improve_count,
    history,
    normalization_mean,
    normalization_std,
    class_weights,
    ema=None,
    save_ema_as_primary=False,
):
    raw_model_state_dict = model.state_dict()
    primary_model_state_dict = (
        ema.model.state_dict()
        if (
            save_ema_as_primary
            and ema is not None
        )
        else raw_model_state_dict
    )

    payload = {
        "epoch": int(epoch),
        "stage": "stage4_physics_multitask",
        "model_name": MODEL_NAME,
        # Keep this key evaluator-compatible. When EMA is the validation model,
        # BEST_CKPT stores EMA weights here so existing evaluators load exactly
        # the model that was selected by validation. LAST_CKPT remains raw.
        "model_state_dict": primary_model_state_dict,
        "raw_model_state_dict": (
            raw_model_state_dict
            if save_ema_as_primary
            else None
        ),
        "primary_weights": (
            "ema"
            if save_ema_as_primary
            else "raw"
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": (
            ema.state_dict()
            if ema is not None
            else None
        ),
        "best_joint_score": float(best_joint_score),
        "best_epoch": int(best_epoch),
        "no_improve_count": int(no_improve_count),
        "validation_protocol": VALIDATION_PROTOCOL_VERSION,
        "validation_csv": str(VAL_CSV),
        "history": history,
        "normalization": {
            "mean": float(normalization_mean),
            "std": float(normalization_std),
        },
        "class_weights": class_weights.detach().cpu().tolist(),
        "loss_weights": {
            "lambda_classification": LAMBDA_CLASSIFICATION,
            "lambda_localization": LAMBDA_LOCALIZATION,
            "lambda_heatmap": LAMBDA_HEATMAP,
            "lambda_offset": LAMBDA_OFFSET,
            "lambda_physics": LAMBDA_PHYSICS,
        },
        "physics_config": {
            "physics_enabled": PHYSICS_ENABLED,
            "profile_direction": PROFILE_DIRECTION,
            "profile_reduction": PROFILE_REDUCTION,
            "profile_preprocess": PROFILE_PREPROCESS,
            "profile_smooth_kernel": PROFILE_SMOOTH_KERNEL,
            "profile_normalization": PROFILE_NORMALIZATION,
            "profile_norm_eps": PROFILE_NORM_EPS,
            "physics_encoder": PHYSICS_ENCODER,
            "physics_dim": PHYSICS_DIM,
            "physics_hidden_channels": list(PHYSICS_HIDDEN_CHANNELS),
            "physics_fusion": PHYSICS_FUSION,
            "physics_fusion_levels": PHYSICS_FUSION_LEVELS,
            "physics_task_routing": PHYSICS_TASK_ROUTING,
            "physics_architecture_version": "v2.2_task_decoupled",
            "physics_alpha_mode": PHYSICS_ALPHA_MODE,
            "physics_alpha_init": PHYSICS_ALPHA_INIT,
            "physics_alpha_max": PHYSICS_ALPHA_MAX,
            "physics_cls_fusion": PHYSICS_CLS_FUSION,
            "physics_cls_alpha_init": PHYSICS_CLS_ALPHA_INIT,
            "physics_cls_alpha_max": PHYSICS_CLS_ALPHA_MAX,
            "use_physics_aux_loss": USE_PHYSICS_AUX_LOSS,
            "init_mode": INIT_MODE,
        },
        "multitask_v2_config": {
            "per_image_heatmap_normalization": (
                USE_PER_IMAGE_HEATMAP_NORMALIZATION
            ),
            "gradient_diagnostics": USE_GRADIENT_DIAGNOSTICS,
            "gradient_diagnostic_interval": GRAD_DIAGNOSTIC_INTERVAL,
            "gradient_diagnostic_scopes": list(GRAD_DIAGNOSTIC_SCOPES),
            "gradient_diagnostic_scope_legacy": GRAD_DIAGNOSTIC_SCOPE,
            "pcgrad": USE_PCGRAD,
            "task_adapters": USE_TASK_ADAPTERS,
            "task_adapter_levels": TASK_ADAPTER_LEVELS,
            "task_adapter_bottleneck_ratio": TASK_ADAPTER_BOTTLENECK_RATIO,
            "differential_lr": USE_DIFFERENTIAL_LR,
            "warm_start_freeze_backbone_epochs": (
                WARM_START_FREEZE_BACKBONE_EPOCHS
            ),
            "ema": USE_EMA,
            "ema_decay": EMA_DECAY,
        },
        "image_width": IMAGE_WIDTH,
        "num_bins": NUM_BINS,
        "bin_width": BIN_WIDTH,
        "primary_tolerance_px": PRIMARY_TOLERANCE_PX,
        "seed": SEED,
    }
    torch.save(payload, path_)


def load_checkpoint(
    path_,
    model,
    optimizer,
    scheduler,
    scaler,
    logger,
    ema=None,
):
    checkpoint = torch.load(path_, map_location=DEVICE)

    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("Checkpoint MODEL_NAME mismatch.")

    checkpoint_stage = checkpoint.get("stage")
    if (
        checkpoint_stage is not None
        and checkpoint_stage != "stage4_physics_multitask"
    ):
        raise ValueError(
            f"Checkpoint stage mismatch: {checkpoint_stage!r}"
        )

    # Bounded alpha stores an unconstrained raw logit under the historical
    # state-dict key "alpha". Therefore an old unbounded Stage-4 checkpoint
    # must NOT be silently resumed into bounded mode, even though tensor names
    # and shapes happen to match. Start a fresh run when changing alpha mode.
    checkpoint_physics = checkpoint.get("physics_config") or {}
    checkpoint_alpha_mode = checkpoint_physics.get("physics_alpha_mode")
    if PHYSICS_ALPHA_MODE == "bounded":
        if checkpoint_alpha_mode != "bounded":
            raise ValueError(
                "Cannot RESUME a checkpoint created before bounded-alpha "
                "parameterization (or with a different alpha mode). "
                "Use RESUME=False and start a fresh controlled run."
            )

    # Routing changes the forward graph even though tensor names/shapes remain
    # compatible. Never resume across routing modes: that would mix two
    # different experiments while silently reusing optimizer state.
    checkpoint_routing = checkpoint_physics.get(
        "physics_task_routing",
        "shared",  # pre-V2.1 checkpoints used shared routing
    )
    if checkpoint_routing != PHYSICS_TASK_ROUTING:
        raise ValueError(
            "Checkpoint PHYSICS_TASK_ROUTING mismatch: "
            f"checkpoint={checkpoint_routing!r}, current={PHYSICS_TASK_ROUTING!r}. "
            "Use RESUME=False and start a fresh routing experiment."
        )

    if PHYSICS_TASK_ROUTING == "dual_task":
        checkpoint_arch = checkpoint_physics.get("physics_architecture_version")
        if checkpoint_arch != "v2.2_task_decoupled":
            raise ValueError(
                "Cannot RESUME dual_task from a pre-V2.2 checkpoint. "
                "Use RESUME=False for the first V2.2 run."
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    ema_state = checkpoint.get("ema_state_dict")
    if ema is not None and ema_state:
        ema.load_state_dict(ema_state)

    start_epoch = int(checkpoint["epoch"]) + 1
    best_joint_score = float(
        checkpoint.get("best_joint_score", -np.inf)
    )
    best_epoch = int(checkpoint.get("best_epoch", 0))
    no_improve_count = int(checkpoint.get("no_improve_count", 0))
    history = checkpoint.get("history", [])

    logger.info(
        "Resumed from %s | next_epoch=%d | "
        "best JointScore=%.6f | best_epoch=%d",
        path_,
        start_epoch,
        best_joint_score,
        best_epoch,
    )

    return (
        start_epoch,
        best_joint_score,
        best_epoch,
        no_improve_count,
        history,
    )


def initialize_from_stage3(
    model,
    logger,
):
    """
    Optional Stage-3 warm start.

    Original V1 paper-primary Stage 4 used INIT_MODE="scratch".
    V2 keeps both modes explicitly selectable so warm-start can be evaluated
    as a controlled training-strategy ablation without deleting V1 behavior.
    """
    if INIT_MODE == "scratch":
        logger.info(
            "Initialization : scratch"
        )
        return

    if INIT_MODE != "stage3":
        raise ValueError(
            f"Unsupported INIT_MODE={INIT_MODE!r}"
        )

    if not STAGE3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Stage-3 checkpoint not found: {STAGE3_CHECKPOINT}"
        )

    checkpoint = torch.load(
        STAGE3_CHECKPOINT,
        map_location=DEVICE,
    )

    if checkpoint.get(
        "model_name"
    ) != MODEL_NAME:
        raise ValueError(
            "Stage-3 checkpoint MODEL_NAME mismatch."
        )

    incompat = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=False,
    )

    unexpected = list(
        incompat.unexpected_keys
    )

    # Missing keys should belong to the new physics branch only.
    non_physics_missing = [
        key
        for key in incompat.missing_keys
        if not (
            key.startswith(
                "physics_encoder."
            )
            or key.startswith(
                "physics_fusions."
            )
            or key.startswith(
                "physics_encoder_cls."
            )
            or key.startswith(
                "physics_cls_fusion."
            )
            or key.startswith(
                "physics_aux_head."
            )
            or key.startswith(
                "classification_adapters."
            )
            or key.startswith(
                "localization_adapters."
            )
        )
    ]

    if unexpected:
        raise RuntimeError(
            "Unexpected Stage-3 checkpoint keys during warm start: "
            f"{unexpected[:20]}"
        )

    if non_physics_missing:
        raise RuntimeError(
            "Non-physics parameters were not restored from Stage 3: "
            f"{non_physics_missing[:20]}"
        )

    logger.info(
        "Initialization : Stage-3 warm start"
    )
    logger.info(
        "Stage-3 ckpt   : %s",
        STAGE3_CHECKPOINT,
    )
    logger.info(
        "New physics params left randomly initialized: %d tensors",
        len(
            incompat.missing_keys
        ),
    )


# =============================================================================
# 11. Metadata / figures
# =============================================================================

def save_run_metadata(
    model,
    total_params,
    trainable_params,
    normalization_mean,
    normalization_std,
    class_weights,
):
    config = {
        "stage": "Stage 4",
        "task": "physics_informed_classification_localization_multitask_v23_val",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "device_policy": "CUDA if available else CPU",
        "actual_device": str(DEVICE),
        "data": {
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "test_csv": str(TEST_CSV),
        },
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "max_epochs": MAX_EPOCHS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "use_differential_lr": USE_DIFFERENTIAL_LR,
            "lr_backbone": LR_BACKBONE,
            "lr_task_heads": LR_TASK_HEADS,
            "lr_physics": LR_PHYSICS,
            "lr_adapters": LR_ADAPTERS,
        },
        "classification": {
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "weighted_cross_entropy": USE_CLASS_WEIGHTS,
            "class_weights": class_weights.detach().cpu().tolist(),
        },
        "localization": {
            "representation": "82-bin Gaussian heatmap + offset",
            "num_bins": NUM_BINS,
            "bin_width": BIN_WIDTH,
            "primary_tolerance_px": PRIMARY_TOLERANCE_PX,
            "class2_localization_masked_in_loss": True,
            "class2_manual_validation_enabled": True,
            "validation_protocol": VALIDATION_PROTOCOL_VERSION,
        },
        "physics": {
            "enabled": PHYSICS_ENABLED,
            "profile_definition_primary": "P(x)=sum_y I(y,x)",
            "profile_direction": PROFILE_DIRECTION,
            "profile_reduction": PROFILE_REDUCTION,
            "weighted_profile_sigma_fraction": (
                WEIGHTED_PROFILE_SIGMA_FRACTION
            ),
            "profile_preprocess": PROFILE_PREPROCESS,
            "profile_smooth_kernel": PROFILE_SMOOTH_KERNEL,
            "profile_normalization": PROFILE_NORMALIZATION,
            "physics_encoder": PHYSICS_ENCODER,
            "physics_dim": PHYSICS_DIM,
            "physics_hidden_channels": list(
                PHYSICS_HIDDEN_CHANNELS
            ),
            "fusion": PHYSICS_FUSION,
            "fusion_levels": PHYSICS_FUSION_LEVELS,
            "task_routing": PHYSICS_TASK_ROUTING,
            "architecture_version": "v2.2_task_decoupled",
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "alpha_mode": PHYSICS_ALPHA_MODE,
            "alpha_init": PHYSICS_ALPHA_INIT,
            "alpha_max": PHYSICS_ALPHA_MAX,
            "classification_late_fusion": PHYSICS_CLS_FUSION,
            "classification_alpha_init": PHYSICS_CLS_ALPHA_INIT,
            "classification_alpha_max": PHYSICS_CLS_ALPHA_MAX,
            "independent_task_physics_encoders": (
                PHYSICS_TASK_ROUTING == "dual_task"
            ),
            "use_auxiliary_physics_loss": USE_PHYSICS_AUX_LOSS,
            "lambda_physics": LAMBDA_PHYSICS,
            "initialization": INIT_MODE,
            "stage3_checkpoint_if_warm_start": str(
                STAGE3_CHECKPOINT
            ),
        },
        "loss": {
            "formula": (
                "lambda_cls*L_cls + "
                "lambda_loc*(lambda_heatmap*L_heatmap + "
                "lambda_offset*L_offset) + "
                "lambda_physics*L_physics"
            ),
            "lambda_classification": LAMBDA_CLASSIFICATION,
            "lambda_localization": LAMBDA_LOCALIZATION,
            "lambda_heatmap": LAMBDA_HEATMAP,
            "lambda_offset": LAMBDA_OFFSET,
            "lambda_physics": LAMBDA_PHYSICS,
            "use_physics_aux_loss": USE_PHYSICS_AUX_LOSS,
            "classification_loss": "WeightedCrossEntropyLoss",
            "heatmap_loss": (
                "PerImageNormalizedModifiedFocalLoss"
                if USE_PER_IMAGE_HEATMAP_NORMALIZATION
                else "CenterNet-style ModifiedFocalLoss (original reduction)"
            ),
            "offset_loss": "SmoothL1Loss",
        },
        "multitask_optimization": {
            "use_gradient_diagnostics": USE_GRADIENT_DIAGNOSTICS,
            "gradient_diagnostic_interval": GRAD_DIAGNOSTIC_INTERVAL,
            "gradient_diagnostic_scopes": list(GRAD_DIAGNOSTIC_SCOPES),
            "gradient_diagnostic_scope_legacy": GRAD_DIAGNOSTIC_SCOPE,
            "use_pcgrad": USE_PCGRAD,
            "use_task_adapters": USE_TASK_ADAPTERS,
            "task_adapter_levels": TASK_ADAPTER_LEVELS,
            "warm_start_freeze_backbone_epochs": (
                WARM_START_FREEZE_BACKBONE_EPOCHS
            ),
            "use_ema": USE_EMA,
            "ema_decay": EMA_DECAY,
            "use_ema_for_validation": USE_EMA_FOR_VALIDATION,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "monitor": "val_joint_score_v23",
            "factor": LR_FACTOR,
            "patience": LR_PATIENCE,
            "min_lr": MIN_LR,
        },
        "checkpoint_selection": {
            "classification_metric": "val Macro-F1",
            "localization_metric_01": "val class-0/1 F1@2px",
            "localization_metric_c2": "val manually annotated class-2 F1@2px",
            "localization_threshold_during_epoch_selection": VAL_CHECKPOINT_THRESHOLD,
            "loc_score": "harmonic_mean(F1_01, F1_class2)",
            "joint_score": "harmonic_mean(classification Macro-F1, LocScore)",
            "validation_protocol": VALIDATION_PROTOCOL_VERSION,
        },
        "normalization": {
            "method": "dataset_zscore",
            "mean": normalization_mean,
            "std": normalization_std,
            "computed_from": str(NORMALIZATION_SOURCE_CSV),
        },
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
        },
        "diagnostics": {
            "epoch_history_csv": str(
                METRICS_DIR
                / "training_history.csv"
            ),
            "batch_loss_csv": str(
                BATCH_LOSS_CSV
            ),
            "fusion_alpha_history_csv": str(
                FUSION_ALPHA_HISTORY_CSV
            ),
            "fusion_alpha_figure": str(
                FUSION_ALPHA_FIGURE
            ),
        },
        "pretrained_weights": False,
        "resize": None,
    }

    with open(OUTPUT_ROOT / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_ROOT / "model.txt", "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")


def save_training_figures(history_df):
    # Total loss
    fig = plt.figure(figsize=(6.4, 4.6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train total", linewidth=1.5)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation total", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Multi-task loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Validation task scores: V2.3-VAL exposes strict and complex localization.
    fig = plt.figure(figsize=(7.2, 5.0))
    plt.plot(
        history_df["epoch"],
        history_df["val_classification_macro_f1"],
        label="Classification Macro-F1",
        linewidth=1.5,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loc01_f1_at_2px"],
        label="Loc-01 F1@2px",
        linewidth=1.3,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loc_c2_f1_at_2px"],
        label="Loc-C2 F1@2px",
        linewidth=1.3,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loc_score"],
        label="LocScore",
        linewidth=1.5,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_joint_score"],
        label="JointScore",
        linewidth=1.5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "val_task_scores.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(matrix):
    fig = plt.figure(figsize=(5.2, 4.6))
    plt.imshow(matrix, interpolation="nearest", aspect="auto")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(NUM_CLASSES), ["0", "1", "2"])
    plt.yticks(range(NUM_CLASSES), ["0", "1", "2"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center"
            )

    plt.tight_layout()
    fig.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 12. Final test outputs
# =============================================================================

def save_final_test_outputs(
    test_result,
    selected_threshold,
    selected_val_metrics,
):
    cls = test_result["classification_metrics"]
    loc = test_result["localization_metrics"]

    payload = {
        "selected_localization_threshold": float(selected_threshold),
        "threshold_selected_from": "val_v2 only; LocScore=H(F1_01,F1_C2)",
        "validation_localization_at_selected_threshold": selected_val_metrics,
        "test_loss": float(test_result["loss"]),
        "test_loss_classification": float(
            test_result["loss_classification"]
        ),
        "test_loss_localization": float(
            test_result["loss_localization"]
        ),
        "test_loss_heatmap": float(test_result["loss_heatmap"]),
        "test_loss_offset": float(test_result["loss_offset"]),
        "test_loss_physics": float(test_result["loss_physics"]),
        "classification": cls,
        "strict_localization_class0_class1": loc,
        "descriptive_test_joint_score": harmonic_joint_score(
            cls["macro_f1"], loc["f1_at_2px"]
        ),
    }

    with open(
        METRICS_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)

    # Classification report
    report = classification_report(
        test_result["y_true"],
        test_result["y_pred"],
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR / "test_classification_report.csv"
    )

    # Confusion matrix
    matrix = confusion_matrix(
        test_result["y_true"],
        test_result["y_pred"],
        labels=list(range(NUM_CLASSES)),
    )
    pd.DataFrame(
        matrix,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    ).to_csv(METRICS_DIR / "confusion_matrix.csv")
    save_confusion_matrix_figure(matrix)

    # Prediction table
    loc_lookup = {
        r["file_name"]: r
        for r in test_result["localization_records"]
    }

    rows = []
    for i, (file_name, true_label, pred_label) in enumerate(
        zip(
            test_result["file_names"],
            test_result["y_true"],
            test_result["y_pred"],
        )
    ):
        row = {
            "file_name": file_name,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "prob_class0": float(test_result["probabilities"][i, 0]),
            "prob_class1": float(test_result["probabilities"][i, 1]),
            "prob_class2": float(test_result["probabilities"][i, 2]),
            "localization_evaluated": int(true_label) in {0, 1},
        }

        if file_name in loc_lookup:
            r = loc_lookup[file_name]
            row.update({
                "gt_position": (
                    r["gt_position"]
                    if r["gt_position"] is not None
                    else np.nan
                ),
                "pred_positions": json.dumps(
                    [round(float(x), 6) for x in r["pred_positions"]]
                ),
                "pred_scores": json.dumps(
                    [round(float(x), 6) for x in r["pred_scores"]]
                ),
                "top1_position": float(r["top1_position"]),
                "top1_score": float(r["top1_score"]),
            })
        else:
            row.update({
                "gt_position": np.nan,
                "pred_positions": "",
                "pred_scores": "",
                "top1_position": np.nan,
                "top1_score": np.nan,
            })

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        METRICS_DIR / "test_predictions.csv",
        index=False,
    )

    # Shared features for later t-SNE / UMAP
    features = test_result["features"]
    if features is not None:
        data = {"file_name": test_result["file_names"]}
        for j in range(features.shape[1]):
            data[f"feature_{j:03d}"] = features[:, j]

        pd.DataFrame(data).to_csv(
            METRICS_DIR / "test_features.csv",
            index=False,
        )


# =============================================================================
# 13. Main
# =============================================================================

def main():
    validate_physics_configuration()
    prepare_output_dirs()
    initialize_diagnostic_files()
    logger = build_logger()
    set_seed(SEED)

    logger.info("=" * 78)
    logger.info(
        "Dark Soliton Stage 4 — "
        "Physics-informed Classification + Localization Multi-task"
    )
    logger.info("=" * 78)

    logger.info("Backbone      : %s", MODEL_NAME)
    logger.info("Device        : %s", DEVICE)
    logger.info("PyTorch       : %s", torch.__version__)
    logger.info("CUDA avail.   : %s", torch.cuda.is_available())

    if MODEL_NAME in {"repvit_m0_9", "fasternet_t0"}:
        timm = detector_base._import_timm()
        logger.info("timm          : %s", timm.__version__)

    logger.info(
        "Loss          : %.2f*L_cls + %.2f*(%.2f*L_heatmap + %.2f*L_offset) "
        "+ %.2f*L_physics",
        LAMBDA_CLASSIFICATION,
        LAMBDA_LOCALIZATION,
        LAMBDA_HEATMAP,
        LAMBDA_OFFSET,
        LAMBDA_PHYSICS,
    )
    logger.info(
        "Physics       : enabled=%s | direction=%s | reduction=%s",
        PHYSICS_ENABLED,
        PROFILE_DIRECTION,
        PROFILE_REDUCTION,
    )
    logger.info(
        "Profile       : preprocess=%s | normalization=%s",
        PROFILE_PREPROCESS,
        PROFILE_NORMALIZATION,
    )
    logger.info(
        "Physics enc.  : %s | dim=%d | hidden=%s",
        PHYSICS_ENCODER,
        PHYSICS_DIM,
        PHYSICS_HIDDEN_CHANNELS,
    )
    if PHYSICS_TASK_ROUTING == "shared":
        cls_route = "spatial physics-fused (shared encoder)"
    elif PHYSICS_TASK_ROUTING == "localization_only":
        cls_route = "visual-only"
    else:
        cls_route = "independent physics global late fusion"
    logger.info(
        "Task routing  : %s | cls=%s | loc=x-aligned spatial physics",
        PHYSICS_TASK_ROUTING,
        cls_route,
    )
    logger.info(
        "Fusion        : %s | levels=%s | alpha_mode=%s | "
        "alpha_init=%.3f | alpha_max=%.3f",
        PHYSICS_FUSION,
        PHYSICS_FUSION_LEVELS,
        PHYSICS_ALPHA_MODE,
        PHYSICS_ALPHA_INIT,
        PHYSICS_ALPHA_MAX,
    )
    if PHYSICS_TASK_ROUTING == "dual_task":
        logger.info(
            "Cls phys fuse : %s | alpha_init=%.3f | alpha_max=%.3f",
            PHYSICS_CLS_FUSION,
            PHYSICS_CLS_ALPHA_INIT,
            PHYSICS_CLS_ALPHA_MAX,
        )
    logger.info(
        "Physics loss  : enabled=%s | lambda=%.3f",
        USE_PHYSICS_AUX_LOSS,
        LAMBDA_PHYSICS,
    )
    logger.info(
        "Init mode     : %s",
        INIT_MODE,
    )
    logger.info(
        "Loss norm     : per-image heatmap=%s",
        USE_PER_IMAGE_HEATMAP_NORMALIZATION,
    )
    logger.info(
        "Grad diag     : enabled=%s | interval=%d | scopes=%s",
        USE_GRADIENT_DIAGNOSTICS,
        GRAD_DIAGNOSTIC_INTERVAL,
        ",".join(GRAD_DIAGNOSTIC_SCOPES),
    )
    logger.info(
        "PCGrad        : %s",
        USE_PCGRAD,
    )
    logger.info(
        "Task adapters : enabled=%s | levels=%s",
        USE_TASK_ADAPTERS,
        TASK_ADAPTER_LEVELS,
    )
    logger.info(
        "Differential LR: %s",
        USE_DIFFERENTIAL_LR,
    )
    logger.info(
        "EMA           : enabled=%s | decay=%.5f | eval=%s",
        USE_EMA,
        EMA_DECAY,
        USE_EMA_FOR_VALIDATION,
    )
    logger.info(
        "Checkpoint    : H(val Macro-F1, LocScore), LocScore=H(F1_01,F1_C2)"
    )
    logger.info(
        "LR scheduler  : ReduceLROnPlateau(mode=max) on V2.3 val JointScore"
    )
    logger.info(
        "Loc threshold : %.2f during checkpoint selection",
        VAL_CHECKPOINT_THRESHOLD,
    )

    # Data
    mean, std = compute_train_normalization(logger)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_datasets_and_loaders(mean, std)

    logger.info(
        "Train: N=%d, counts=%s",
        len(train_dataset),
        train_dataset.class_counts(),
    )
    logger.info(
        "Val  : N=%d, counts=%s",
        len(val_dataset),
        val_dataset.class_counts(),
    )
    val_v2_summary = validate_val_v2_metadata(val_dataset)
    logger.info(
        "Val-v2 class2  : total=%d | eligible=%d | valid=%d | no-reliable=%d | "
        "excluded=%d | GT=%d",
        val_v2_summary["class2_total"],
        val_v2_summary["class2_eligible"],
        val_v2_summary["class2_valid_positive"],
        val_v2_summary["class2_no_reliable"],
        val_v2_summary["class2_excluded"],
        val_v2_summary["class2_total_gt"],
    )
    logger.info(
        "Test : N=%d, counts=%s",
        len(test_dataset),
        test_dataset.class_counts(),
    )

    # Model
    model = PhysicsInformedMultiTaskModel(
        MODEL_NAME,
        NUM_CLASSES,
    ).to(DEVICE)

    initialize_from_stage3(
        model,
        logger,
    )

    log_physics_shape_diagnostic(
        model,
        train_loader,
        logger,
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(
        "Parameters    : total=%s, trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # Losses
    class_weights = compute_class_weights(train_dataset)
    logger.info(
        "Class weights : %s",
        [round(float(x), 6) for x in class_weights.detach().cpu().tolist()],
    )

    classification_criterion = nn.CrossEntropyLoss(
        weight=class_weights if USE_CLASS_WEIGHTS else None
    )

    if USE_PER_IMAGE_HEATMAP_NORMALIZATION:
        heatmap_criterion = (
            PerImageNormalizedModifiedFocalLoss(
                alpha=detector_base.FOCAL_ALPHA,
                beta=detector_base.FOCAL_BETA,
            )
        )
    else:
        # Original V1 / Stage-2 loss reduction retained for controlled ablation.
        heatmap_criterion = detector_base.ModifiedFocalLoss(
            alpha=detector_base.FOCAL_ALPHA,
            beta=detector_base.FOCAL_BETA,
        )

    offset_criterion = nn.SmoothL1Loss()

    optimizer = build_optimizer(model)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
        and not USE_PCGRAD
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )
    logger.info("AMP enabled   : %s", amp_enabled)
    if USE_PCGRAD and USE_AMP and DEVICE.type == "cuda":
        logger.info(
            "AMP requested but disabled because PCGrad uses explicit "
            "unscaled task gradients."
        )

    ema = (
        ModelEMA(model, decay=EMA_DECAY)
        if USE_EMA
        else None
    )

    save_run_metadata(
        model,
        total_params,
        trainable_params,
        mean,
        std,
        class_weights,
    )

    # Resume state
    start_epoch = 1
    best_joint_score = -np.inf
    best_epoch = 0
    no_improve_count = 0
    history = []

    if RESUME:
        if not LAST_CKPT.exists():
            raise FileNotFoundError(
                f"RESUME=True but checkpoint not found: {LAST_CKPT}"
            )

        (
            start_epoch,
            best_joint_score,
            best_epoch,
            no_improve_count,
            history,
        ) = load_checkpoint(
            LAST_CKPT,
            model,
            optimizer,
            scheduler,
            scaler,
            logger,
            ema=ema,
        )

    # Training
    training_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()

        maybe_update_backbone_freeze_state(
            model,
            epoch,
            logger,
        )

        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            amp_enabled,
            classification_criterion,
            heatmap_criterion,
            offset_criterion,
            epoch,
            ema=ema,
        )

        append_batch_loss_records(
            train_result[
                "batch_records"
            ]
        )
        append_gradient_diagnostic_records(
            train_result[
                "gradient_records"
            ]
        )
        save_gradient_diagnostics_epoch_summary()

        eval_model = (
            ema.model
            if (
                ema is not None
                and USE_EMA_FOR_VALIDATION
            )
            else model
        )

        val_result = evaluate_model(
            eval_model,
            val_loader,
            classification_criterion,
            heatmap_criterion,
            offset_criterion,
            VAL_CHECKPOINT_THRESHOLD,
            collect_features=False,
        )

        val_cls = val_result["classification_metrics"]
        val_loc = val_result["localization_metrics_01"]
        val_loc_c2 = val_result["localization_metrics_c2"]
        if val_loc_c2 is None:
            raise RuntimeError(
                "V2.3-VAL requires class-2 manual localization metrics, but none were produced."
            )
        loc_score = float(val_result["localization_score"])
        joint = float(val_result["joint_score"])

        # Scheduler, checkpoint selection and early stopping now optimize the
        # same validation objective.
        scheduler.step(joint)
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_values = optimizer_lr_dict(optimizer)

        alpha_values = (
            get_fusion_alpha_values(
                model
            )
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            **lr_values,

            "train_loss": train_result["loss"],
            "train_loss_classification": (
                train_result["loss_classification"]
            ),
            "train_loss_localization": (
                train_result["loss_localization"]
            ),
            "train_loss_heatmap": train_result["loss_heatmap"],
            "train_loss_offset": train_result["loss_offset"],
            "train_loss_physics": train_result["loss_physics"],

            "val_loss": val_result["loss"],
            "val_loss_classification": (
                val_result["loss_classification"]
            ),
            "val_loss_localization": (
                val_result["loss_localization"]
            ),
            "val_loss_heatmap": val_result["loss_heatmap"],
            "val_loss_offset": val_result["loss_offset"],
            "val_loss_physics": val_result["loss_physics"],

            "val_classification_accuracy": val_cls["accuracy"],
            "val_classification_balanced_accuracy": (
                val_cls["balanced_accuracy"]
            ),
            "val_classification_macro_f1": val_cls["macro_f1"],

            "val_localization_precision_at_2px": (
                val_loc["precision_at_2px"]
            ),
            "val_localization_recall_at_2px": (
                val_loc["recall_at_2px"]
            ),
            "val_localization_f1_at_2px": (
                val_loc["f1_at_2px"]
            ),
            "val_localization_acc_at_1px": (
                val_loc["acc_at_1px"]
            ),
            "val_localization_acc_at_2px": (
                val_loc["acc_at_2px"]
            ),
            "val_localization_acc_at_3px": (
                val_loc["acc_at_3px"]
            ),
            "val_localization_top1_mae_px": (
                val_loc["top1_mae_px"]
            ),
            "val_localization_matched_mae_px": (
                val_loc["matched_mae_px"]
            ),

            # V2.3-VAL explicit dual-localization validation metrics.
            "val_loc01_precision_at_2px": val_loc["precision_at_2px"],
            "val_loc01_recall_at_2px": val_loc["recall_at_2px"],
            "val_loc01_f1_at_2px": val_loc["f1_at_2px"],
            "val_loc01_matched_mae_px": val_loc["matched_mae_px"],

            "val_loc_c2_precision_at_2px": val_loc_c2["precision_at_2px"],
            "val_loc_c2_recall_at_2px": val_loc_c2["recall_at_2px"],
            "val_loc_c2_f1_at_2px": val_loc_c2["f1_at_2px"],
            "val_loc_c2_matched_mae_px": val_loc_c2["matched_mae_px"],
            "val_loc_c2_exact_count_accuracy": val_loc_c2["exact_count_accuracy"],
            "val_loc_c2_exact_sample_success_at_2px": val_loc_c2["exact_sample_success_at_2px"],
            "val_loc_c2_count_mae": val_loc_c2["count_mae"],
            "val_loc_c2_false_positives_per_image": val_loc_c2["false_positives_per_image"],
            "val_loc_c2_false_negatives_per_image": val_loc_c2["false_negatives_per_image"],
            "val_loc_c2_num_images": val_loc_c2["num_images"],
            "val_loc_c2_total_gt_excitations": val_loc_c2["total_gt_excitations"],
            "val_loc_score": loc_score,

            "grad_diagnostic_samples": len(train_result["gradient_records"]),

            "backbone_grad_cosine_mean": _mean_gradient_record_field(
                train_result["gradient_records"], "backbone_grad_cosine"
            ),
            "backbone_grad_negative_cosine_rate": _mean_gradient_record_field(
                train_result["gradient_records"], "backbone_gradient_conflict"
            ),
            "backbone_grad_norm_ratio_other_over_cls_mean": _mean_gradient_record_field(
                train_result["gradient_records"],
                "backbone_grad_norm_ratio_other_over_cls",
            ),

            "physics_grad_cosine_mean": _mean_gradient_record_field(
                train_result["gradient_records"], "physics_grad_cosine"
            ),
            "physics_grad_negative_cosine_rate": _mean_gradient_record_field(
                train_result["gradient_records"], "physics_gradient_conflict"
            ),
            "physics_grad_norm_ratio_other_over_cls_mean": _mean_gradient_record_field(
                train_result["gradient_records"],
                "physics_grad_norm_ratio_other_over_cls",
            ),

            "shared_grad_cosine_mean": _mean_gradient_record_field(
                train_result["gradient_records"], "shared_grad_cosine"
            ),
            "shared_grad_negative_cosine_rate": _mean_gradient_record_field(
                train_result["gradient_records"], "shared_gradient_conflict"
            ),
            "shared_grad_norm_ratio_other_over_cls_mean": _mean_gradient_record_field(
                train_result["gradient_records"],
                "shared_grad_norm_ratio_other_over_cls",
            ),

            # Legacy aliases = shared path.
            "grad_cosine_mean": _mean_gradient_record_field(
                train_result["gradient_records"], "shared_grad_cosine"
            ),
            "grad_negative_cosine_rate": _mean_gradient_record_field(
                train_result["gradient_records"], "shared_gradient_conflict"
            ),
            "grad_norm_ratio_other_over_cls_mean": _mean_gradient_record_field(
                train_result["gradient_records"],
                "shared_grad_norm_ratio_other_over_cls",
            ),

            "val_joint_score": joint,
            "epoch_seconds": time.time() - epoch_start,

            **alpha_values,
        }

        # V2.2.1 FIX:
        # Populate epoch-level gradient summaries for every configured scope.
        # The previous V2.2 row only materialized backbone/physics/shared,
        # while the logger iterated over physics_cls/physics_loc as well,
        # which caused KeyError after epoch 1.
        for scope in GRAD_DIAGNOSTIC_SCOPES:
            row[f"{scope}_grad_cosine_mean"] = _mean_gradient_record_field(
                train_result["gradient_records"],
                f"{scope}_grad_cosine",
            )
            row[f"{scope}_grad_negative_cosine_rate"] = _mean_gradient_record_field(
                train_result["gradient_records"],
                f"{scope}_gradient_conflict",
            )
            row[f"{scope}_grad_norm_ratio_other_over_cls_mean"] = (
                _mean_gradient_record_field(
                    train_result["gradient_records"],
                    f"{scope}_grad_norm_ratio_other_over_cls",
                )
            )
            row[f"{scope}_grad_cls_norm_mean"] = _mean_gradient_record_field(
                train_result["gradient_records"],
                f"{scope}_grad_cls_norm",
            )
            row[f"{scope}_grad_other_norm_mean"] = _mean_gradient_record_field(
                train_result["gradient_records"],
                f"{scope}_grad_other_norm",
            )

        history.append(row)

        improved = joint > best_joint_score + MIN_DELTA

        if improved:
            best_joint_score = joint
            best_epoch = epoch
            no_improve_count = 0

            save_checkpoint(
                BEST_CKPT,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_joint_score,
                best_epoch,
                no_improve_count,
                history,
                mean,
                std,
                class_weights,
                ema=ema,
                save_ema_as_primary=(
                    ema is not None
                    and USE_EMA_FOR_VALIDATION
                ),
            )
        else:
            no_improve_count += 1

        save_checkpoint(
            LAST_CKPT,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_joint_score,
            best_epoch,
            no_improve_count,
            history,
            mean,
            std,
            class_weights,
            ema=ema,
            save_ema_as_primary=False,
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            METRICS_DIR / "training_history.csv",
            index=False,
        )

        save_fusion_alpha_history(
            history_df
        )

        marker = " *BEST*" if improved else ""

        # V2.2.1 FIX:
        # Terminal output should show only effective bounded alphas.
        # Raw alpha logits (e.g. -1.028) are internal unconstrained parameters
        # and are intentionally excluded from the human-readable log.
        alpha_parts = []

        cls_alpha_value = alpha_values.get(
            "physics_cls_alpha",
            float("nan"),
        )
        if np.isfinite(cls_alpha_value):
            alpha_parts.append(
                f"cls={cls_alpha_value:.3f}"
            )

        for key, value in sorted(alpha_values.items()):
            if (
                key.startswith("physics_alpha_level")
                and np.isfinite(value)
            ):
                level_name = key.replace(
                    "physics_alpha_level",
                    "locL",
                )
                alpha_parts.append(
                    f"{level_name}={value:.3f}"
                )

        alpha_log = (
            " / ".join(alpha_parts)
            if alpha_parts
            else "none"
        )

        logger.info(
            "Epoch %03d/%03d | lr=%.2e | "
            "train=%.5f (cls=%.5f,loc=%.5f,hm=%.5f,off=%.5f,phys=%.5f) | "
            "val=%.5f | ClsF1=%.4f | Loc01F1=%.4f | LocC2F1=%.4f | "
            "LocScore=%.4f | Joint=%.4f | alpha=%s | %.1fs%s",
            epoch,
            MAX_EPOCHS,
            current_lr,
            train_result["loss"],
            train_result["loss_classification"],
            train_result["loss_localization"],
            train_result["loss_heatmap"],
            train_result["loss_offset"],
            train_result["loss_physics"],
            val_result["loss"],
            val_cls["macro_f1"],
            val_loc["f1_at_2px"],
            val_loc_c2["f1_at_2px"],
            loc_score,
            joint,
            alpha_log,
            row["epoch_seconds"],
            marker,
        )

        if train_result["gradient_records"]:
            for scope in GRAD_DIAGNOSTIC_SCOPES:
                logger.info(
                    "Grad %-8s | cos=% .4f | conflict_rate=%.3f | "
                    "norm(other/cls)=%.3f | samples=%d",
                    scope,
                    row.get(
                        f"{scope}_grad_cosine_mean",
                        float("nan"),
                    ),
                    row.get(
                        f"{scope}_grad_negative_cosine_rate",
                        float("nan"),
                    ),
                    row.get(
                        f"{scope}_grad_norm_ratio_other_over_cls_mean",
                        float("nan"),
                    ),
                    row["grad_diagnostic_samples"],
                )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping at epoch %d: JointScore did not improve "
                "by > %.6f for %d epochs.",
                epoch,
                MIN_DELTA,
                EARLY_STOP_PATIENCE,
            )
            break

    # History / figures
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        METRICS_DIR / "training_history.csv",
        index=False,
    )
    save_training_figures(
        history_df
    )
    save_fusion_alpha_history(
        history_df
    )

    logger.info(
        "Batch losses    : %s",
        BATCH_LOSS_CSV,
    )
    logger.info(
        "Alpha history   : %s",
        FUSION_ALPHA_HISTORY_CSV,
    )
    logger.info(
        "Alpha figure    : %s",
        FUSION_ALPHA_FIGURE,
    )
    if USE_GRADIENT_DIAGNOSTICS:
        logger.info(
            "Grad diagnostics: %s",
            GRAD_DIAGNOSTICS_CSV,
        )
        logger.info(
            "Grad epoch summary: %s",
            GRAD_DIAGNOSTICS_EPOCH_CSV,
        )

    logger.info(
        "Training finished | %.2f min",
        (time.time() - training_start) / 60.0,
    )
    logger.info(
        "Best epoch=%d | best V2.3 val JointScore=%.6f",
        best_epoch,
        best_joint_score,
    )

    # Load best
    if not BEST_CKPT.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CKPT}"
        )

    best_checkpoint = torch.load(
        BEST_CKPT, map_location=DEVICE
    )
    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )
    logger.info(
        "Loaded best checkpoint primary weights: %s",
        best_checkpoint.get("primary_weights", "raw"),
    )

    # Validation forward once -> cached threshold search
    best_val_result = evaluate_model(
        model,
        val_loader,
        classification_criterion,
        heatmap_criterion,
        offset_criterion,
        VAL_CHECKPOINT_THRESHOLD,
        collect_features=False,
    )

    selected_threshold, selected_val_bundle, _ = (
        search_best_localization_threshold(
            best_val_result["localization_cache_01"],
            best_val_result["localization_cache_c2"],
        )
    )
    selected_val_loc01 = selected_val_bundle["loc01"]
    selected_val_c2 = selected_val_bundle["loc_c2"]
    selected_val_loc_score = selected_val_bundle["loc_score"]

    logger.info(
        "Selected val-v2 threshold=%.2f | Loc01 F1=%.4f | LocC2 F1=%.4f | "
        "LocScore=%.4f | C2 P/R=%.4f/%.4f",
        selected_threshold,
        selected_val_loc01["f1_at_2px"],
        selected_val_c2["f1_at_2px"],
        selected_val_loc_score,
        selected_val_c2["precision_at_2px"],
        selected_val_c2["recall_at_2px"],
    )

    # Final test
    test_result = evaluate_model(
        model,
        test_loader,
        classification_criterion,
        heatmap_criterion,
        offset_criterion,
        selected_threshold,
        collect_features=True,
    )

    save_final_test_outputs(
        test_result,
        selected_threshold,
        selected_val_bundle,
    )

    test_cls = test_result["classification_metrics"]
    test_loc = test_result["localization_metrics"]

    logger.info("=" * 78)
    logger.info("FINAL TEST RESULTS — BEST PHYSICS-INFORMED MULTI-TASK CHECKPOINT")
    logger.info("=" * 78)

    logger.info("Classification:")
    logger.info("  Accuracy               : %.6f", test_cls["accuracy"])
    logger.info(
        "  Balanced Accuracy      : %.6f",
        test_cls["balanced_accuracy"],
    )
    logger.info(
        "  Macro Precision        : %.6f",
        test_cls["macro_precision"],
    )
    logger.info(
        "  Macro Recall           : %.6f",
        test_cls["macro_recall"],
    )
    logger.info("  Macro F1               : %.6f", test_cls["macro_f1"])
    logger.info(
        "  Weighted F1            : %.6f",
        test_cls["weighted_f1"],
    )

    logger.info("Strict localization (class-0/1 only):")
    logger.info("  Threshold              : %.2f", selected_threshold)
    logger.info(
        "  Precision @ 2 px       : %.6f",
        test_loc["precision_at_2px"],
    )
    logger.info(
        "  Recall @ 2 px          : %.6f",
        test_loc["recall_at_2px"],
    )
    logger.info(
        "  F1 @ 2 px              : %.6f",
        test_loc["f1_at_2px"],
    )
    logger.info(
        "  Acc @ 1 px             : %.6f",
        test_loc["acc_at_1px"],
    )
    logger.info(
        "  Acc @ 2 px             : %.6f",
        test_loc["acc_at_2px"],
    )
    logger.info(
        "  Acc @ 3 px             : %.6f",
        test_loc["acc_at_3px"],
    )
    logger.info(
        "  Top-1 MAE              : %.6f px",
        test_loc["top1_mae_px"],
    )
    logger.info(
        "  Matched MAE            : %.6f px",
        test_loc["matched_mae_px"],
    )
    logger.info(
        "  FP / image             : %.6f",
        test_loc["false_positives_per_image"],
    )

    logger.info(
        "Descriptive test JointScore: %.6f",
        harmonic_joint_score(
            test_cls["macro_f1"],
            test_loc["f1_at_2px"],
        ),
    )
    logger.info("Output directory         : %s", OUTPUT_ROOT)

    if test_result["features"] is not None:
        logger.info(
            "Test shared features     : shape=%s -> %s",
            tuple(test_result["features"].shape),
            METRICS_DIR / "test_features.csv",
        )

    logger.info("=" * 78)


if __name__ == "__main__":
    main()
