# !/usr/bin/env python3
# -*- coding: utf-8 -*-
###########flchen 2026.11 ###############
"""
evaluate_gold_challenge_stage4.py
=================================

Stage 4 Physics-informed Classification + Localization multi-task 模型
在统一人工 Gold 数据集上的最终泛化测试。
"""

from pathlib import Path
from functools import lru_cache
import ast
import json
import logging
import sys
import importlib

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import train_detector_baseline as detector_module
import train_physics_multitask_v23 as stage4_module
# -----------------------------------------------------------------------------
# Stage-4 training module loader
# -----------------------------------------------------------------------------
# V2.2 introduces two extra modules in the checkpoint:
#   physics_encoder_cls.*
#   physics_cls_fusion.*
# Therefore the Gold evaluator MUST construct the V2.2 model definition before
# strict state-dict loading.  Loading the legacy train_physics_multitask.py
# silently creates the old architecture and necessarily causes Unexpected keys.
_STAGE4_MODULE_CANDIDATES = (
    "train_physics_multitask_v23",     # common descriptive filename
    "train_physics_multitask_v22",     # accepted if V2.3 was saved under old name
    "train_physics_multitask",         # accepted only if it contains dual-task V2.2+ architecture
)


def _load_stage4_training_module():
    import_errors = []

    for module_name in _STAGE4_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue

        # Do not accidentally accept the legacy Stage-4 module.
        is_v22 = (
            hasattr(module, "PHYSICS_TASK_ROUTING")
            and hasattr(module, "PhysicsClassificationLateFusion")
            and hasattr(module, "PhysicsInformedMultiTaskModel")
        )

        if is_v22:
            return module, module_name

        import_errors.append(
            f"{module_name}: imported successfully but it is an older "
            "Stage-4 architecture (missing V2.2 dual-task classes)."
        )

    details = "\n  - ".join(import_errors)
    raise ImportError(
        "Cannot load the Stage-4 V2.3 training module.\n"
        "Place the V2.3 training script in the project root, preferably as\n"
        "    train_physics_multitask_v23.py\n"
        "The evaluator intentionally refuses the legacy model because strict=False "
        "would ignore the trained classification-physics branch.\n"
        f"Tried:\n  - {details}"
    )


stage4_module, STAGE4_TRAIN_MODULE_NAME = _load_stage4_training_module()

from dark_soliton_dataloader import DarkSolitonDataset, make_dataloader


# =============================================================================
# 1. Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

GOLD_CSV = DATA_DIR / "class2_class9_inference2_manual_annotations.csv"

# MODEL_NAME = "simple_cnn"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
MODEL_NAME = "fasternet_t0"

ANNOTATOR = None

BATCH_SIZE = 32
NUM_WORKERS = 2
PRIMARY_TOLERANCE_PX = 2.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_OUTPUT_ROOT = (
    ROOT / "output" / MODEL_NAME / "physics_multitask"
)

CHECKPOINT_PATH = (
    MODEL_OUTPUT_ROOT / "checkpoint" / "best.pt.tar"
)

TRAIN_CONFIG_JSON = (
    MODEL_OUTPUT_ROOT / "config.json"
)

VAL_THRESHOLD_CSV = (
    MODEL_OUTPUT_ROOT / "metrics" / "val_threshold_search.csv"
)

OUTPUT_DIR = (
    MODEL_OUTPUT_ROOT / "gold_challenge_combined"
)

PREDICTIONS_CSV = OUTPUT_DIR / "challenge_predictions.csv"
LOCALIZATION_METRICS_JSON = OUTPUT_DIR / "challenge_metrics.json"
CLASSIFICATION_METRICS_JSON = OUTPUT_DIR / "classification_metrics.json"
PER_SOURCE_CSV = OUTPUT_DIR / "per_source_metrics.csv"
PER_COUNT_CSV = OUTPUT_DIR / "per_count_metrics.csv"
ANNOTATION_SUMMARY_JSON = OUTPUT_DIR / "annotation_summary.json"
CONFUSION_COMBINED_CSV = OUTPUT_DIR / "classification_confusion_combined.csv"
CONFUSION_CLASS9_CSV = OUTPUT_DIR / "classification_confusion_class9_inference2.csv"

# Paper-oriented separated outputs
CLASS2_LOCALIZATION_JSON = OUTPUT_DIR / "class2_localization_metrics.json"
CLASS9_LOCALIZATION_JSON = OUTPUT_DIR / "class9_localization_metrics.json"
CLASS9_CLASSIFICATION_JSON = OUTPUT_DIR / "class9_classification_metrics.json"
CLASS2_CLASSIFICATION_JSON = OUTPUT_DIR / "class2_classification_descriptive.json"
CLASS9_CLASSIFICATION_REPORT_CSV = OUTPUT_DIR / "class9_classification_report.csv"
PAPER_CORE_JSON = OUTPUT_DIR / "paper_core_metrics.json"
PAPER_CORE_CSV = OUTPUT_DIR / "paper_core_metrics.csv"

PHYSICS_CONFIG_JSON = OUTPUT_DIR / "physics_config_used.json"
FUSION_ALPHA_JSON = OUTPUT_DIR / "learned_fusion_alpha.json"
FUSION_ALPHA_CSV = OUTPUT_DIR / "learned_fusion_alpha.csv"

# Physics-fused shared Gold features for t-SNE / UMAP.
GOLD_FEATURES_CSV = OUTPUT_DIR / "gold_features.csv"


# =============================================================================
# 2. Helpers
# =============================================================================

def build_logger():
    logger = logging.getLogger("stage4_gold_challenge")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def parse_position_list(value):
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except Exception:
            raw = ast.literal_eval(text)

    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"excitation_positions is not a list: {value!r}"
        )

    return sorted(float(x) for x in raw)


def source_group_from_row(row):
    source_csv = str(row.get("source_csv", "")).strip()

    if "class2_test_challenge" in source_csv:
        return "class2_test_gold"

    if "class9_inference2" in source_csv:
        return "class9_inference2_gold"

    if source_csv:
        return Path(source_csv).stem

    source_split = str(row.get("source_split", "")).strip()
    return source_split if source_split else "unknown"



def load_training_config():
    """
    Read Stage-4 config.json if available.

    Checkpoint physics_config is authoritative for the saved checkpoint.
    config.json fills extra settings not stored in the checkpoint, such as
    physics_hidden_channels and residual-profile preprocessing details.
    """
    if not TRAIN_CONFIG_JSON.exists():
        return None

    with open(
        TRAIN_CONFIG_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    config_model_name = config.get(
        "model_name"
    )

    if (
        config_model_name is not None
        and config_model_name != MODEL_NAME
    ):
        raise ValueError(
            "Stage-4 config model mismatch: "
            f"{config_model_name!r} != {MODEL_NAME!r}"
        )

    return config


def apply_stage4_physics_configuration(
    checkpoint,
    training_config,
):
    """
    Restore architecture-critical Stage-4 / V2.2 globals BEFORE constructing
    PhysicsInformedMultiTaskModel.

    Why this is required for V2.2
    -----------------------------
    V2.2 adds an independent classification physics encoder and late-fusion
    module.  The model class only creates these modules when
    PHYSICS_TASK_ROUTING == "dual_task".  Therefore the checkpoint's routing,
    bounded-alpha settings, and classification-fusion settings must be restored
    before model construction.
    """
    checkpoint_physics = checkpoint.get("physics_config")

    if not checkpoint_physics:
        raise KeyError(
            "Stage-4 checkpoint has no physics_config. "
            "Use a checkpoint generated by the Stage-4 training script."
        )

    file_physics = {}
    if training_config is not None:
        file_physics = dict(training_config.get("physics", {}))

    # Check checkpoint/config overlap.  The checkpoint remains authoritative.
    overlap_pairs = {
        "physics_enabled": "enabled",
        "profile_direction": "profile_direction",
        "profile_reduction": "profile_reduction",
        "profile_preprocess": "profile_preprocess",
        "profile_normalization": "profile_normalization",
        "physics_encoder": "physics_encoder",
        "physics_dim": "physics_dim",
        "physics_fusion": "fusion",
        "physics_fusion_levels": "fusion_levels",
        "physics_task_routing": "task_routing",
        "physics_architecture_version": "architecture_version",
        "physics_alpha_mode": "alpha_mode",
        "physics_alpha_init": "alpha_init",
        "physics_alpha_max": "alpha_max",
        "physics_cls_fusion": "classification_late_fusion",
        "physics_cls_alpha_init": "classification_alpha_init",
        "physics_cls_alpha_max": "classification_alpha_max",
        "use_physics_aux_loss": "use_auxiliary_physics_loss",
        "init_mode": "initialization",
    }

    for ckpt_key, cfg_key in overlap_pairs.items():
        if ckpt_key in checkpoint_physics and cfg_key in file_physics:
            ckpt_value = checkpoint_physics[ckpt_key]
            cfg_value = file_physics[cfg_key]
            if ckpt_value != cfg_value:
                raise ValueError(
                    "Checkpoint/config physics mismatch: "
                    f"{ckpt_key}={ckpt_value!r} vs "
                    f"{cfg_key}={cfg_value!r}"
                )

    effective = {}

    def set_module_attr(attr_name, value):
        if value is None:
            return
        if not hasattr(stage4_module, attr_name):
            raise AttributeError(
                f"Loaded Stage-4 module {STAGE4_TRAIN_MODULE_NAME!r} does not "
                f"define {attr_name}. This evaluator requires the V2.2 "
                "training module used to create the checkpoint."
            )
        setattr(stage4_module, attr_name, value)
        effective[attr_name] = value

    def ckpt_value(key, cast=None):
        if key not in checkpoint_physics:
            return None
        value = checkpoint_physics[key]
        return cast(value) if cast is not None else value

    # Shared Stage-4 settings.
    set_module_attr("PHYSICS_ENABLED", ckpt_value("physics_enabled", bool))
    set_module_attr("PROFILE_DIRECTION", ckpt_value("profile_direction"))
    set_module_attr("PROFILE_REDUCTION", ckpt_value("profile_reduction"))
    set_module_attr("PROFILE_PREPROCESS", ckpt_value("profile_preprocess"))
    set_module_attr("PROFILE_NORMALIZATION", ckpt_value("profile_normalization"))
    set_module_attr("PHYSICS_ENCODER", ckpt_value("physics_encoder"))
    set_module_attr("PHYSICS_DIM", ckpt_value("physics_dim", int))
    set_module_attr("PHYSICS_FUSION", ckpt_value("physics_fusion"))
    set_module_attr("PHYSICS_FUSION_LEVELS", ckpt_value("physics_fusion_levels"))
    set_module_attr("PHYSICS_ALPHA_INIT", ckpt_value("physics_alpha_init", float))
    set_module_attr("USE_PHYSICS_AUX_LOSS", ckpt_value("use_physics_aux_loss", bool))
    set_module_attr("INIT_MODE", ckpt_value("init_mode"))

    # V2.2 architecture-critical settings.
    routing = ckpt_value("physics_task_routing")
    architecture_version = ckpt_value("physics_architecture_version")

    if routing is None:
        # Old checkpoints can still be evaluated by an old evaluator, but this
        # V2.2 evaluator should not guess the architecture.
        raise KeyError(
            "Checkpoint physics_config has no 'physics_task_routing'. "
            "This does not look like a V2.2 checkpoint."
        )

    set_module_attr("PHYSICS_TASK_ROUTING", routing)
    set_module_attr("PHYSICS_ALPHA_MODE", ckpt_value("physics_alpha_mode"))
    set_module_attr("PHYSICS_ALPHA_MAX", ckpt_value("physics_alpha_max", float))
    set_module_attr("PHYSICS_CLS_FUSION", ckpt_value("physics_cls_fusion"))
    set_module_attr("PHYSICS_CLS_ALPHA_INIT", ckpt_value("physics_cls_alpha_init", float))
    set_module_attr("PHYSICS_CLS_ALPHA_MAX", ckpt_value("physics_cls_alpha_max", float))

    # These are also stored directly in the V2.2 checkpoint; prefer them to
    # config.json so the model is reconstructed from the checkpoint itself.
    set_module_attr("PROFILE_SMOOTH_KERNEL", ckpt_value("profile_smooth_kernel", int))
    set_module_attr("PROFILE_NORM_EPS", ckpt_value("profile_norm_eps", float))

    hidden_channels = checkpoint_physics.get("physics_hidden_channels")
    if hidden_channels is not None:
        set_module_attr(
            "PHYSICS_HIDDEN_CHANNELS",
            tuple(int(v) for v in hidden_channels),
        )

    # Weighted-profile sigma was historically stored in config.json only.
    if "weighted_profile_sigma_fraction" in file_physics:
        set_module_attr(
            "WEIGHTED_PROFILE_SIGMA_FRACTION",
            float(file_physics["weighted_profile_sigma_fraction"]),
        )

    # Restore architecture-critical MTL switches too.  They are OFF in the
    # current V2.2 experiment, but this makes the evaluator safe for later
    # V2.2 checkpoints that enable task adapters.
    v2_config = checkpoint.get("multitask_v2_config", {}) or {}
    if "task_adapters" in v2_config:
        set_module_attr("USE_TASK_ADAPTERS", bool(v2_config["task_adapters"]))
    if "task_adapter_levels" in v2_config:
        set_module_attr("TASK_ADAPTER_LEVELS", v2_config["task_adapter_levels"])
    if "task_adapter_bottleneck_ratio" in v2_config:
        set_module_attr(
            "TASK_ADAPTER_BOTTLENECK_RATIO",
            float(v2_config["task_adapter_bottleneck_ratio"]),
        )

    loss_weights = checkpoint.get("loss_weights", {}) or {}
    if "lambda_physics" in loss_weights:
        set_module_attr("LAMBDA_PHYSICS", float(loss_weights["lambda_physics"]))

    stage4_module.MODEL_NAME = MODEL_NAME
    stage4_module.validate_physics_configuration()

    # Strong safeguard: V2.2 dual-task checkpoints must be reconstructed with
    # the classification physics path present.  Never use strict=False here.
    if routing == "dual_task":
        if not hasattr(stage4_module, "PhysicsClassificationLateFusion"):
            raise RuntimeError(
                "The loaded Stage-4 module is not V2.2: "
                "PhysicsClassificationLateFusion is missing."
            )

    return {
        "stage4_train_module": STAGE4_TRAIN_MODULE_NAME,
        "checkpoint_physics_config": checkpoint_physics,
        "training_config_physics": file_physics,
        "effective_module_config": effective,
        "physics_architecture_version": architecture_version,
    }


def extract_learned_fusion_alpha(model):
    """Extract EFFECTIVE bounded alpha values for both V2.2 physics paths."""
    rows = []

    # Localization / spatial physics path.
    fusion_modules = getattr(model, "physics_fusions", None)
    if fusion_modules is not None:
        for level_index, module in enumerate(fusion_modules):
            module_type = type(module).__name__
            feature_channels = getattr(module, "feature_channels", None)
            fusion_type = getattr(module, "fusion_type", "identity")

            alpha_value = None
            alpha = getattr(module, "alpha", None)
            effective_fn = getattr(module, "effective_alpha", None)

            if callable(effective_fn):
                alpha_tensor = effective_fn()
                alpha_value = float(alpha_tensor.detach().cpu().item())
            elif alpha is not None:
                alpha_value = float(alpha.detach().cpu().item())

            rows.append({
                "branch": "localization_spatial",
                "level_index": int(level_index),
                "module_type": module_type,
                "fusion_type": str(fusion_type),
                "feature_channels": (
                    None if feature_channels is None else int(feature_channels)
                ),
                "alpha_mode": getattr(stage4_module, "PHYSICS_ALPHA_MODE", None),
                "alpha_init": float(stage4_module.PHYSICS_ALPHA_INIT),
                "alpha_max": float(getattr(stage4_module, "PHYSICS_ALPHA_MAX", np.nan)),
                "alpha_learned": alpha_value,
                "delta_from_init": (
                    None if alpha_value is None
                    else float(alpha_value - stage4_module.PHYSICS_ALPHA_INIT)
                ),
            })

    # V2.2 classification / global late-fusion physics path.
    cls_fusion = getattr(model, "physics_cls_fusion", None)
    if cls_fusion is not None:
        alpha_value = None
        alpha = getattr(cls_fusion, "alpha", None)
        effective_fn = getattr(cls_fusion, "effective_alpha", None)

        if callable(effective_fn):
            alpha_tensor = effective_fn()
            alpha_value = float(alpha_tensor.detach().cpu().item())
        elif alpha is not None:
            alpha_value = float(alpha.detach().cpu().item())

        rows.append({
            "branch": "classification_global",
            "level_index": -1,
            "module_type": type(cls_fusion).__name__,
            "fusion_type": str(getattr(cls_fusion, "fusion_type", "unknown")),
            "feature_channels": int(
                getattr(cls_fusion, "visual_channels", 0) or 0
            ),
            "alpha_mode": getattr(stage4_module, "PHYSICS_ALPHA_MODE", None),
            "alpha_init": float(stage4_module.PHYSICS_CLS_ALPHA_INIT),
            "alpha_max": float(stage4_module.PHYSICS_CLS_ALPHA_MAX),
            "alpha_learned": alpha_value,
            "delta_from_init": (
                None if alpha_value is None
                else float(alpha_value - stage4_module.PHYSICS_CLS_ALPHA_INIT)
            ),
        })

    return rows


def save_stage4_physics_diagnostics(
    model,
    applied_config,
):
    save_json(
        applied_config,
        PHYSICS_CONFIG_JSON,
    )

    alpha_rows = (
        extract_learned_fusion_alpha(
            model
        )
    )

    save_json(
        alpha_rows,
        FUSION_ALPHA_JSON,
    )

    pd.DataFrame(
        alpha_rows
    ).to_csv(
        FUSION_ALPHA_CSV,
        index=False,
    )

    return alpha_rows


# =============================================================================
# 3. Gold annotation loading / validation
# =============================================================================

def load_and_validate_gold():
    if not GOLD_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find Gold CSV: {GOLD_CSV}"
        )

    df = pd.read_csv(GOLD_CSV)

    required_columns = {
        "file_name",
        "relative_path",
        "source_split",
        "source_csv",
        "num_excitations",
        "excitation_positions",
        "annotation_status",
        "confidence",
        "manual_label",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            f"Gold CSV missing columns: {sorted(missing)}"
        )

    if (
        ANNOTATOR is not None
        and "annotator" in df.columns
    ):
        df = df[
            df["annotator"].astype(str) == str(ANNOTATOR)
        ].copy()

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Gold CSV is empty after filtering.")

    duplicate = df["file_name"].astype(str).duplicated(keep=False)
    if duplicate.any():
        examples = (
            df.loc[duplicate, "file_name"]
            .astype(str)
            .tolist()[:10]
        )
        raise ValueError(
            "Gold CSV contains duplicated file_name values. "
            "Use a single consensus/annotator table first. "
            f"Examples: {examples}"
        )

    df["manual_label"] = pd.to_numeric(
        df["manual_label"],
        errors="raise",
    ).astype(int)

    invalid_labels = sorted(
        set(df["manual_label"].tolist()) - {-1, 0, 1, 2}
    )
    if invalid_labels:
        raise ValueError(
            f"Invalid manual_label values: {invalid_labels}"
        )

    parsed_positions = []
    source_groups = []

    for _, row in df.iterrows():
        status = str(row["annotation_status"]).strip()
        manual_label = int(row["manual_label"])
        positions = parse_position_list(
            row["excitation_positions"]
        )
        num_excitations = int(row["num_excitations"])

        for position in positions:
            if not (0.0 <= position <= 163.0):
                raise ValueError(
                    f"{row['file_name']}: position={position} outside [0,163]"
                )

        if status == "valid":
            if num_excitations != len(positions):
                raise ValueError(
                    f"{row['file_name']}: num_excitations != len(positions)"
                )
            if num_excitations <= 0:
                raise ValueError(
                    f"{row['file_name']}: valid sample needs >=1 position"
                )
            if manual_label in {-1, 0}:
                raise ValueError(
                    f"{row['file_name']}: valid localization sample "
                    f"cannot have manual_label={manual_label}"
                )
            if manual_label == 1 and num_excitations != 1:
                raise ValueError(
                    f"{row['file_name']}: manual_label=1 must have exactly 1 position"
                )

        elif status == "no_reliable_soliton":
            if num_excitations != 0 or len(positions) != 0:
                raise ValueError(
                    f"{row['file_name']}: no_reliable_soliton must have 0 GT"
                )
            if manual_label not in {0, 2}:
                raise ValueError(
                    f"{row['file_name']}: no_reliable_soliton currently "
                    f"expects manual_label 0 or 2, got {manual_label}"
                )

        elif status == "ambiguous":
            if len(positions) != 0:
                raise ValueError(
                    f"{row['file_name']}: ambiguous sample should keep no reliable positions"
                )

        else:
            raise ValueError(
                f"{row['file_name']}: unsupported annotation_status={status!r}"
            )

        parsed_positions.append(positions)
        source_groups.append(source_group_from_row(row))

    df["_gt_positions"] = parsed_positions
    df["_source_group"] = source_groups

    df["_included_localization_primary"] = (
        df["annotation_status"]
        .astype(str)
        .isin(["valid", "no_reliable_soliton"])
    )

    # Classification Gold is defined by manual_label itself.
    # Important:
    # - original class-2 localization-ambiguous samples can still be class-2
    #   semantically and therefore remain valid for classification;
    # - class-9 semantic ambiguous samples use manual_label=-1 and are excluded.
    df["_included_classification"] = (
        df["manual_label"].isin([0, 1, 2])
    )

    return df


# =============================================================================
# 4. Validation-only threshold
# =============================================================================

def select_validation_threshold():
    if not VAL_THRESHOLD_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find Stage-4 val threshold CSV: {VAL_THRESHOLD_CSV}"
        )

    df = pd.read_csv(VAL_THRESHOLD_CSV)

    required = {
        "threshold",
        "precision_at_2px",
        "f1_at_2px",
        "false_positives_per_image",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"val_threshold_search.csv missing: {sorted(missing)}"
        )

    ranked = df.sort_values(
        by=[
            "f1_at_2px",
            "precision_at_2px",
            "false_positives_per_image",
            "threshold",
        ],
        ascending=[False, False, True, False],
        kind="mergesort",
    )

    best = ranked.iloc[0]

    return float(best["threshold"]), best.to_dict()


# =============================================================================
# 5. One-to-one matching
# =============================================================================

def match_positions(gt_positions, pred_positions, tolerance_px):
    gt = tuple(sorted(float(x) for x in gt_positions))
    pred = tuple(sorted(float(x) for x in pred_positions))

    @lru_cache(maxsize=None)
    def solve(i, j):
        if i >= len(gt) or j >= len(pred):
            return 0, 0.0, ()

        candidates = [
            solve(i + 1, j),
            solve(i, j + 1),
        ]

        error = abs(gt[i] - pred[j])

        if error <= tolerance_px:
            sub_count, sub_error, sub_pairs = solve(i + 1, j + 1)
            candidates.append(
                (
                    sub_count + 1,
                    sub_error + error,
                    ((gt[i], pred[j], error),) + sub_pairs,
                )
            )

        return sorted(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )[0]

    _, _, pairs = solve(0, 0)

    return [
        {
            "gt": float(g),
            "pred": float(p),
            "error": float(e),
        }
        for g, p, e in pairs
    ]


# =============================================================================
# 6. Localization metrics
# =============================================================================

def calculate_localization_metrics(records, tolerance_px=2.0):
    included = [
        r for r in records
        if r["included_primary"]
    ]

    if not included:
        return None

    total_gt = 0
    total_pred = 0
    total_tp = 0
    matched_errors = []
    exact_count_hits = 0
    exact_sample_hits = 0
    count_abs_errors = []

    for record in included:
        gt_positions = record["gt_positions"]
        pred_positions = record["pred_positions"]

        pairs = match_positions(
            gt_positions,
            pred_positions,
            tolerance_px,
        )

        n_gt = len(gt_positions)
        n_pred = len(pred_positions)
        n_tp = len(pairs)

        total_gt += n_gt
        total_pred += n_pred
        total_tp += n_tp

        matched_errors.extend(
            pair["error"] for pair in pairs
        )

        exact_count = n_pred == n_gt
        exact_sample = exact_count and n_tp == n_gt

        exact_count_hits += int(exact_count)
        exact_sample_hits += int(exact_sample)
        count_abs_errors.append(abs(n_pred - n_gt))

    fp = total_pred - total_tp
    fn = total_gt - total_tp

    precision = (
        total_tp / (total_tp + fp)
        if (total_tp + fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + fn)
        if (total_tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "num_images": int(len(included)),
        "total_gt_excitations": int(total_gt),
        "total_predictions": int(total_pred),
        "true_positives": int(total_tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision_at_2px": float(precision),
        "recall_at_2px": float(recall),
        "f1_at_2px": float(f1),
        "matched_mae_at_2px": (
            float(np.mean(matched_errors))
            if matched_errors
            else float("nan")
        ),
        "exact_count_accuracy": float(
            exact_count_hits / len(included)
        ),
        "count_mae": float(
            np.mean(count_abs_errors)
        ),
        "exact_sample_success_at_2px": float(
            exact_sample_hits / len(included)
        ),
        "false_positives_per_image": float(
            fp / len(included)
        ),
        "false_negatives_per_image": float(
            fn / len(included)
        ),
    }


def calculate_gt_match_rate(records, tolerance_px):
    included = [
        r for r in records
        if r["included_primary"]
    ]

    total_gt = 0
    total_matched = 0

    for record in included:
        total_gt += len(record["gt_positions"])
        total_matched += len(
            match_positions(
                record["gt_positions"],
                record["pred_positions"],
                tolerance_px,
            )
        )

    if total_gt == 0:
        return float("nan")

    return float(total_matched / total_gt)


def metric_bundle(records):
    metrics = calculate_localization_metrics(
        records,
        PRIMARY_TOLERANCE_PX,
    )

    if metrics is None:
        return None

    metrics["gt_match_rate_at_1px"] = (
        calculate_gt_match_rate(records, 1.0)
    )
    metrics["gt_match_rate_at_2px"] = (
        calculate_gt_match_rate(records, 2.0)
    )
    metrics["gt_match_rate_at_3px"] = (
        calculate_gt_match_rate(records, 3.0)
    )

    return metrics


# =============================================================================
# 7. Classification metrics
# =============================================================================

def classification_metrics_from_records(records):
    subset = [
        r for r in records
        if r["included_classification"]
    ]

    if not subset:
        return None

    y_true = np.asarray(
        [r["manual_label"] for r in subset],
        dtype=int,
    )
    y_pred = np.asarray(
        [r["pred_class"] for r in subset],
        dtype=int,
    )

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    weighted_f1 = float(
        np.average(
            f1,
            weights=support,
        )
    ) if support.sum() > 0 else float("nan")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    return {
        "num_images": int(len(subset)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in [0, 1, 2]
        },
        "confusion_matrix": cm.tolist(),
    }


def class2_descriptive_metrics(records):
    """
    Class-2 classification is a single-class descriptive subset.
    Therefore Accuracy / mean probabilities / predicted-class distribution
    are meaningful, while 3-class Balanced Accuracy / Macro-F1 are not used
    as primary metrics for this subset alone.
    """
    subset = [
        r for r in records
        if r["manual_label"] == 2
        and r["included_classification"]
    ]

    if not subset:
        return None

    pred_classes = np.asarray(
        [r["pred_class"] for r in subset],
        dtype=int,
    )

    prob0 = np.asarray([r["prob_class0"] for r in subset], dtype=float)
    prob1 = np.asarray([r["prob_class1"] for r in subset], dtype=float)
    prob2 = np.asarray([r["prob_class2"] for r in subset], dtype=float)

    counts = {
        int(label): int(np.sum(pred_classes == label))
        for label in [0, 1, 2]
    }

    n = len(subset)

    return {
        "num_images": int(n),
        "class2_accuracy": float(np.mean(pred_classes == 2)),
        "mean_prob_class0": float(np.mean(prob0)),
        "mean_prob_class1": float(np.mean(prob1)),
        "mean_prob_class2": float(np.mean(prob2)),
        "predicted_class_counts": {
            str(label): counts[label]
            for label in [0, 1, 2]
        },
        "predicted_class_rates": {
            str(label): float(counts[label] / n)
            for label in [0, 1, 2]
        },
    }


def save_confusion_csv(metrics, path):
    if metrics is None:
        return

    cm = np.asarray(
        metrics["confusion_matrix"],
        dtype=int,
    )

    pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    ).to_csv(path)


def save_classification_report_from_metrics(metrics, path):
    if metrics is None:
        return

    rows = []
    for label in [0, 1, 2]:
        m = metrics["per_class"][str(label)]
        rows.append({
            "scope": f"class_{label}",
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "support": m["support"],
        })

    rows.append({
        "scope": "macro",
        "precision": metrics["macro_precision"],
        "recall": metrics["macro_recall"],
        "f1": metrics["macro_f1"],
        "support": metrics["num_images"],
    })
    rows.append({
        "scope": "weighted",
        "precision": np.nan,
        "recall": np.nan,
        "f1": metrics["weighted_f1"],
        "support": metrics["num_images"],
    })

    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(payload, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )


def log_full_localization_metrics(logger, title, metrics, paper_primary=False):
    if metrics is None:
        logger.info("%s: no eligible localization samples.", title)
        return

    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)

    prefix = "[PAPER CORE] " if paper_primary else "[AUX] "

    logger.info("%sPrecision @ 2 px           : %.6f", prefix, metrics["precision_at_2px"])
    logger.info("%sRecall @ 2 px              : %.6f", prefix, metrics["recall_at_2px"])
    logger.info("%sF1 @ 2 px                  : %.6f", prefix, metrics["f1_at_2px"])
    logger.info("%sMatched MAE @ 2 px         : %.6f px", prefix, metrics["matched_mae_at_2px"])
    logger.info("%sExact Count Accuracy       : %.6f", prefix, metrics["exact_count_accuracy"])
    logger.info("%sExact Sample Success @ 2px : %.6f", prefix, metrics["exact_sample_success_at_2px"])

    logger.info("-" * 78)
    logger.info("[AUX] Eligible images        : %d", metrics["num_images"])
    logger.info("[AUX] Total GT excitations   : %d", metrics["total_gt_excitations"])
    logger.info("[AUX] Total predictions      : %d", metrics["total_predictions"])
    logger.info("[AUX] TP / FP / FN           : %d / %d / %d", metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"])
    logger.info("[AUX] GT Match Rate @ 1 px   : %.6f", metrics["gt_match_rate_at_1px"])
    logger.info("[AUX] GT Match Rate @ 2 px   : %.6f", metrics["gt_match_rate_at_2px"])
    logger.info("[AUX] GT Match Rate @ 3 px   : %.6f", metrics["gt_match_rate_at_3px"])
    logger.info("[AUX] Count MAE              : %.6f", metrics["count_mae"])
    logger.info("[AUX] FP / image             : %.6f", metrics["false_positives_per_image"])
    logger.info("[AUX] FN / image             : %.6f", metrics["false_negatives_per_image"])


def log_full_classification_metrics(logger, title, metrics, paper_primary=False):
    if metrics is None:
        logger.info("%s: no eligible classification samples.", title)
        return

    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)

    if paper_primary:
        logger.info("[PAPER CORE] Accuracy          : %.6f", metrics["accuracy"])
        logger.info("[PAPER CORE] Balanced Accuracy : %.6f", metrics["balanced_accuracy"])
        logger.info("[PAPER CORE] Macro F1          : %.6f", metrics["macro_f1"])
    else:
        logger.info("[AUX] Accuracy                 : %.6f", metrics["accuracy"])
        logger.info("[AUX] Balanced Accuracy        : %.6f", metrics["balanced_accuracy"])
        logger.info("[AUX] Macro F1                 : %.6f", metrics["macro_f1"])

    logger.info("-" * 78)
    logger.info("[AUX] N                        : %d", metrics["num_images"])
    logger.info("[AUX] Macro Precision          : %.6f", metrics["macro_precision"])
    logger.info("[AUX] Macro Recall             : %.6f", metrics["macro_recall"])
    logger.info("[AUX] Weighted F1              : %.6f", metrics["weighted_f1"])

    for label in [0, 1, 2]:
        m = metrics["per_class"][str(label)]
        logger.info(
            "[AUX] class-%d | P=%.6f R=%.6f F1=%.6f N=%d",
            label,
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
        )



def save_gold_feature_table(
    feature_records,
    output_csv,
):
    """
    Save one row per Gold sample.

    Feature columns:
        feature_000, feature_001, ...

    Metadata columns are kept together with the feature vector so that
    downstream t-SNE / UMAP code can filter the exact Gold subset without
    rejoining another table.
    """
    if not feature_records:
        raise RuntimeError(
            "No Gold features were collected."
        )

    feature_df = pd.DataFrame(
        feature_records
    )

    feature_columns = [
        column
        for column in feature_df.columns
        if column.startswith("feature_")
    ]

    if not feature_columns:
        raise RuntimeError(
            "Gold feature table contains no feature_* columns."
        )

    feature_df.to_csv(
        output_csv,
        index=False,
    )

    return (
        len(feature_df),
        len(feature_columns),
    )


# =============================================================================
# 8. Inference
# =============================================================================

@torch.no_grad()
def run_evaluation(model, loader, gold_df, threshold):
    model.eval()
    records = []
    feature_records = []

    for batch in loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        outputs = model(images)

        class_probs = (
            torch.softmax(
                outputs["classification_logits"],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        heatmap_logits = outputs["heatmap_logits"]
        offset_logits = outputs["offset_logits"]

        # Physics-informed model returns the pooled deepest FUSED feature
        # immediately before the classification FC layer.
        shared_features = (
            outputs["shared_feature"]
            .detach()
            .cpu()
            .numpy()
        )

        indices = batch["index"].cpu().tolist()

        for batch_index, row_index in enumerate(indices):
            row = gold_df.iloc[int(row_index)]

            decoded = detector_module.decode_single_prediction(
                heatmap_logits=heatmap_logits[batch_index],
                offset_logits=offset_logits[batch_index],
                threshold=threshold,
            )

            included_loc = bool(
                row["_included_localization_primary"]
            )

            included_cls = bool(
                row["_included_classification"]
            )

            probs = class_probs[batch_index]

            records.append(
                {
                    "file_name": str(row["file_name"]),
                    "source_csv": str(row["source_csv"]),
                    "source_split": str(row["source_split"]),
                    "source_group": str(row["_source_group"]),
                    "manual_label": int(row["manual_label"]),
                    "annotation_status": str(row["annotation_status"]),
                    "confidence": str(row.get("confidence", "")),
                    "included_primary": included_loc,
                    "included_classification": included_cls,
                    "gt_positions": (
                        list(row["_gt_positions"])
                        if included_loc
                        else []
                    ),
                    "pred_positions": decoded["pred_positions"],
                    "pred_scores": decoded["pred_scores"],
                    "pred_class": int(np.argmax(probs)),
                    "prob_class0": float(probs[0]),
                    "prob_class1": float(probs[1]),
                    "prob_class2": float(probs[2]),
                }
            )

            feature_vector = (
                shared_features[
                    batch_index
                ]
                .reshape(-1)
            )

            feature_row = {
                "file_name": str(row["file_name"]),
                "source_csv": str(row["source_csv"]),
                "source_split": str(row["source_split"]),
                "source_group": str(row["_source_group"]),
                "manual_label": int(row["manual_label"]),
                "annotation_status": str(
                    row["annotation_status"]
                ),
                "included_primary": included_loc,
                "included_classification": included_cls,
                "pred_class": int(np.argmax(probs)),
                "feature_type": (
                    "v2.2_task_decoupled_classification_feature_before_fc"
                ),
            }

            feature_row.update(
                {
                    f"feature_{i:03d}": float(value)
                    for i, value
                    in enumerate(feature_vector)
                }
            )

            feature_records.append(
                feature_row
            )

    return records, feature_records


# =============================================================================
# 9. Save outputs
# =============================================================================

def save_prediction_table(records):
    rows = []

    for record in records:
        gt_positions = record["gt_positions"]
        pred_positions = record["pred_positions"]

        if record["included_primary"]:
            pairs = match_positions(
                gt_positions,
                pred_positions,
                PRIMARY_TOLERANCE_PX,
            )

            tp = len(pairs)
            fp = len(pred_positions) - tp
            fn = len(gt_positions) - tp
            exact_count = (
                len(pred_positions) == len(gt_positions)
            )
            exact_sample = (
                exact_count and tp == len(gt_positions)
            )
        else:
            pairs = []
            tp = np.nan
            fp = np.nan
            fn = np.nan
            exact_count = np.nan
            exact_sample = np.nan

        rows.append(
            {
                "file_name": record["file_name"],
                "source_group": record["source_group"],
                "source_csv": record["source_csv"],
                "source_split": record["source_split"],
                "manual_label": record["manual_label"],
                "annotation_status": record["annotation_status"],
                "confidence": record["confidence"],
                "included_localization_primary": record["included_primary"],
                "included_classification": record["included_classification"],

                "pred_class": record["pred_class"],
                "prob_class0": record["prob_class0"],
                "prob_class1": record["prob_class1"],
                "prob_class2": record["prob_class2"],

                "gt_count": (
                    len(gt_positions)
                    if record["included_primary"]
                    else np.nan
                ),
                "gt_positions": json.dumps(
                    [round(float(x), 6) for x in gt_positions]
                ),
                "pred_count": len(pred_positions),
                "pred_positions": json.dumps(
                    [round(float(x), 6) for x in pred_positions]
                ),
                "pred_scores": json.dumps(
                    [round(float(x), 6) for x in record["pred_scores"]]
                ),
                "matched_pairs_at_2px": (
                    json.dumps(
                        [
                            {
                                "gt": round(pair["gt"], 6),
                                "pred": round(pair["pred"], 6),
                                "error": round(pair["error"], 6),
                            }
                            for pair in pairs
                        ]
                    )
                    if record["included_primary"]
                    else ""
                ),
                "tp_at_2px": tp,
                "fp_at_2px": fp,
                "fn_at_2px": fn,
                "exact_count": exact_count,
                "exact_sample_success_at_2px": exact_sample,
            }
        )

    pd.DataFrame(rows).to_csv(
        PREDICTIONS_CSV,
        index=False,
    )


def build_per_source_table(records):
    rows = []

    groups = ["combined"] + sorted(
        set(r["source_group"] for r in records)
    )

    for group in groups:
        subset = (
            records
            if group == "combined"
            else [
                r for r in records
                if r["source_group"] == group
            ]
        )

        metrics = metric_bundle(subset)
        if metrics is None:
            continue

        rows.append(
            {
                "source_group": group,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def build_per_count_table(records):
    rows = []

    source_groups = ["combined"] + sorted(
        set(r["source_group"] for r in records)
    )

    for source_group in source_groups:
        subset = (
            records
            if source_group == "combined"
            else [
                r for r in records
                if r["source_group"] == source_group
            ]
        )

        primary = [
            r for r in subset
            if r["included_primary"]
        ]

        gt_counts = sorted(
            set(
                len(r["gt_positions"])
                for r in primary
            )
        )

        for gt_count in gt_counts:
            count_subset = [
                r for r in primary
                if len(r["gt_positions"]) == gt_count
            ]

            metrics = metric_bundle(count_subset)
            if metrics is None:
                continue

            rows.append(
                {
                    "source_group": source_group,
                    "gt_count": int(gt_count),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 10. Main
# =============================================================================

def main():
    logger = build_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info("STAGE-4 PHYSICS-INFORMED MANUAL GOLD CHALLENGE")
    logger.info("=" * 78)
    logger.info("Model          : %s", MODEL_NAME)
    logger.info("Device         : %s", DEVICE)
    logger.info("Checkpoint     : %s", CHECKPOINT_PATH)
    logger.info("Train config   : %s", TRAIN_CONFIG_JSON)
    logger.info("Train module   : %s", STAGE4_TRAIN_MODULE_NAME)

    gold_df = load_and_validate_gold()

    status_counts = (
        gold_df["annotation_status"]
        .value_counts()
        .to_dict()
    )

    label_counts = (
        gold_df["manual_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    source_counts = (
        gold_df["_source_group"]
        .value_counts()
        .to_dict()
    )

    primary_df = gold_df[
        gold_df["_included_localization_primary"]
    ]

    classification_df = gold_df[
        gold_df["_included_classification"]
    ]

    total_primary_gt = int(
        primary_df["_gt_positions"]
        .map(len)
        .sum()
    )

    annotation_summary = {
        "total_rows": int(len(gold_df)),
        "status_counts": {
            str(k): int(v)
            for k, v in status_counts.items()
        },
        "manual_label_counts": {
            str(k): int(v)
            for k, v in label_counts.items()
        },
        "source_group_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },
        "localization_primary_rows": int(len(primary_df)),
        "localization_primary_total_gt": total_primary_gt,
        "classification_gold_rows": int(len(classification_df)),
        "localization_primary_statuses": [
            "valid",
            "no_reliable_soliton",
        ],
        "localization_excluded_statuses": [
            "ambiguous",
        ],
        "classification_included_manual_labels": [
            0,
            1,
            2,
        ],
        "classification_excluded_manual_labels": [
            -1,
        ],
    }

    with open(
        ANNOTATION_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            annotation_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Gold rows       : %d", len(gold_df))
    logger.info("Status counts   : %s", status_counts)
    logger.info("Manual labels   : %s", label_counts)
    logger.info("Source groups   : %s", source_counts)
    logger.info(
        "Primary loc     : %d images / %d GT excitations",
        len(primary_df),
        total_primary_gt,
    )
    logger.info(
        "Classification  : %d non-ambiguous labels",
        len(classification_df),
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Stage-4 best checkpoint not found: {CHECKPOINT_PATH}"
        )

    threshold, threshold_row = select_validation_threshold()

    logger.info(
        "Threshold       : %.2f (Stage-4 validation only)",
        threshold,
    )
    logger.info(
        "Val F1@2       : %.6f",
        float(threshold_row["f1_at_2px"]),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    checkpoint_protocol = checkpoint.get("validation_protocol")
    if (
        checkpoint_protocol is not None
        and checkpoint_protocol != "v2.3_val_aware"
    ):
        raise ValueError(
            "Checkpoint validation protocol mismatch: "
            f"{checkpoint_protocol!r} != 'v2.3_val_aware'."
        )

    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != MODEL_NAME:
        raise ValueError(
            f"Checkpoint model mismatch: "
            f"{checkpoint_model_name!r} != {MODEL_NAME!r}"
        )

    checkpoint_stage = checkpoint.get("stage")
    if (
        checkpoint_stage is not None
        and checkpoint_stage != "stage4_physics_multitask"
    ):
        raise ValueError(
            f"Checkpoint stage mismatch: {checkpoint_stage!r}"
        )

    training_config = load_training_config()

    applied_physics_config = (
        apply_stage4_physics_configuration(
            checkpoint=checkpoint,
            training_config=training_config,
        )
    )

    logger.info(
        "Physics config : routing=%s | direction=%s | reduction=%s | "
        "preprocess=%s | encoder=%s | loc_fusion=%s/%s | "
        "cls_fusion=%s | alpha_mode=%s",
        stage4_module.PHYSICS_TASK_ROUTING,
        stage4_module.PROFILE_DIRECTION,
        stage4_module.PROFILE_REDUCTION,
        stage4_module.PROFILE_PREPROCESS,
        stage4_module.PHYSICS_ENCODER,
        stage4_module.PHYSICS_FUSION,
        stage4_module.PHYSICS_FUSION_LEVELS,
        stage4_module.PHYSICS_CLS_FUSION,
        stage4_module.PHYSICS_ALPHA_MODE,
    )

    normalization = checkpoint.get("normalization")
    if not normalization:
        raise KeyError(
            "Checkpoint has no normalization information."
        )

    mean = float(normalization["mean"])
    std = float(normalization["std"])

    dataset = DarkSolitonDataset(
        csv_path=GOLD_CSV,
        task="unlabeled",
        normalize="dataset_zscore",
        dataset_mean=mean,
        dataset_std=std,
        check_file_exists=True,
    )

    if len(dataset) != len(gold_df):
        raise RuntimeError(
            "Dataset row count != validated Gold row count. "
            "Use a single consensus/annotator Gold CSV."
        )

    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = stage4_module.PhysicsInformedMultiTaskModel(
        model_name=MODEL_NAME,
        num_classes=3,
    ).to(DEVICE)

    # V2.2 checkpoint/model architecture sanity check BEFORE strict loading.
    state_dict = checkpoint["model_state_dict"]
    has_cls_physics_weights = any(
        key.startswith("physics_encoder_cls.")
        or key.startswith("physics_cls_fusion.")
        for key in state_dict
    )
    model_has_cls_physics = (
        getattr(model, "physics_encoder_cls", None) is not None
        and getattr(model, "physics_cls_fusion", None) is not None
    )

    if has_cls_physics_weights and not model_has_cls_physics:
        raise RuntimeError(
            "V2.2 checkpoint contains classification-physics weights, but the "
            "constructed model has no classification physics branch. "
            "Check that the V2.3 training module (s4v23.py or train_physics_multitask_v23.py) is present and that "
            "PHYSICS_TASK_ROUTING was restored as 'dual_task'. "
            "Do NOT change strict=True to strict=False."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    alpha_rows = save_stage4_physics_diagnostics(
        model=model,
        applied_config=applied_physics_config,
    )

    if alpha_rows:
        logger.info("Learned EFFECTIVE fusion alpha:")
        for row in alpha_rows:
            if row["alpha_learned"] is None:
                logger.info(
                    "  branch=%s | level=%d | fusion=%s | alpha=N/A",
                    row["branch"],
                    row["level_index"],
                    row["fusion_type"],
                )
            else:
                logger.info(
                    "  branch=%s | level=%d | fusion=%s | "
                    "alpha_init=%.6f | alpha=%.6f | alpha_max=%.6f | delta=%+.6f",
                    row["branch"],
                    row["level_index"],
                    row["fusion_type"],
                    row["alpha_init"],
                    row["alpha_learned"],
                    row["alpha_max"],
                    row["delta_from_init"],
                )

    (
        records,
        feature_records,
    ) = run_evaluation(
        model=model,
        loader=loader,
        gold_df=gold_df,
        threshold=threshold,
    )

    feature_n, feature_dim = (
        save_gold_feature_table(
            feature_records,
            GOLD_FEATURES_CSV,
        )
    )

    logger.info(
        "Gold features   : N=%d, D=%d -> %s",
        feature_n,
        feature_dim,
        GOLD_FEATURES_CSV,
    )

    save_prediction_table(records)

    # -----------------------------------------------------------------
    # Localization
    # -----------------------------------------------------------------
    combined_loc_metrics = metric_bundle(records)

    per_source_df = build_per_source_table(records)
    per_source_df.to_csv(
        PER_SOURCE_CSV,
        index=False,
    )

    per_count_df = build_per_count_table(records)
    per_count_df.to_csv(
        PER_COUNT_CSV,
        index=False,
    )

    by_source = {}
    for _, row in per_source_df.iterrows():
        group = str(row["source_group"])
        by_source[group] = {
            key: (
                None
                if pd.isna(value)
                else (
                    int(value)
                    if key in {
                        "num_images",
                        "total_gt_excitations",
                        "total_predictions",
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    }
                    else float(value)
                )
            )
            for key, value in row.items()
            if key != "source_group"
        }

    localization_output = {
        "stage": "stage4_physics_multitask",
        "model_name": MODEL_NAME,
        "checkpoint": str(CHECKPOINT_PATH),
        "gold_csv": str(GOLD_CSV),
        "threshold": float(threshold),
        "threshold_source": str(VAL_THRESHOLD_CSV),
        "primary_tolerance_px": PRIMARY_TOLERANCE_PX,
        "physics_config_file": str(PHYSICS_CONFIG_JSON),
        "fusion_alpha_file": str(FUSION_ALPHA_CSV),
        "combined": combined_loc_metrics,
        "by_source": by_source,
    }

    with open(
        LOCALIZATION_METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            localization_output,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    # -----------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------
    combined_cls = classification_metrics_from_records(
        records
    )

    class9_records = [
        r for r in records
        if r["source_group"] == "class9_inference2_gold"
    ]

    class2_records = [
        r for r in records
        if r["source_group"] == "class2_test_gold"
    ]

    class9_cls = classification_metrics_from_records(
        class9_records
    )

    class2_desc = class2_descriptive_metrics(
        class2_records
    )

    # -------------------------------------------------------------
    # Paper-oriented separated tasks
    # -------------------------------------------------------------
    # Paper-primary localization result: class-2 Gold challenge.
    class2_loc_metrics = metric_bundle(class2_records)

    # Auxiliary localization result: class-9 Gold.
    class9_loc_metrics = metric_bundle(class9_records)

    save_json(class2_loc_metrics, CLASS2_LOCALIZATION_JSON)
    save_json(class9_loc_metrics, CLASS9_LOCALIZATION_JSON)
    save_json(class9_cls, CLASS9_CLASSIFICATION_JSON)
    save_json(class2_desc, CLASS2_CLASSIFICATION_JSON)

    save_classification_report_from_metrics(
        class9_cls,
        CLASS9_CLASSIFICATION_REPORT_CSV,
    )

    paper_core = {
        "stage": "stage4_physics_multitask",
        "model_name": MODEL_NAME,
        "gold_csv": str(GOLD_CSV),
        "physics_config_file": str(PHYSICS_CONFIG_JSON),
        "fusion_alpha_file": str(FUSION_ALPHA_CSV),
        "paper_primary_tasks": {
            "class2_localization": {
                "subset": "class2_test_gold",
                "eligible_images": None if class2_loc_metrics is None else class2_loc_metrics["num_images"],
                "precision_at_2px": None if class2_loc_metrics is None else class2_loc_metrics["precision_at_2px"],
                "recall_at_2px": None if class2_loc_metrics is None else class2_loc_metrics["recall_at_2px"],
                "f1_at_2px": None if class2_loc_metrics is None else class2_loc_metrics["f1_at_2px"],
                "matched_mae_at_2px": None if class2_loc_metrics is None else class2_loc_metrics["matched_mae_at_2px"],
                "exact_count_accuracy": None if class2_loc_metrics is None else class2_loc_metrics["exact_count_accuracy"],
                "exact_sample_success_at_2px": None if class2_loc_metrics is None else class2_loc_metrics["exact_sample_success_at_2px"],
            },
            "class9_classification": {
                "subset": "class9_inference2_gold",
                "eligible_images": None if class9_cls is None else class9_cls["num_images"],
                "accuracy": None if class9_cls is None else class9_cls["accuracy"],
                "balanced_accuracy": None if class9_cls is None else class9_cls["balanced_accuracy"],
                "macro_f1": None if class9_cls is None else class9_cls["macro_f1"],
            },
        },
        "note": (
            "Recommended paper-primary comparison: Stage2/Stage3/Stage4 "
            "on class-2 localization, and Stage1/Stage3/Stage4 on held-out "
            "class-9 classification. Combined metrics are retained only "
            "for audit/reference."
        ),
    }

    save_json(paper_core, PAPER_CORE_JSON)

    paper_rows = []
    if class2_loc_metrics is not None:
        for metric_name in [
            "precision_at_2px",
            "recall_at_2px",
            "f1_at_2px",
            "matched_mae_at_2px",
            "exact_count_accuracy",
            "exact_sample_success_at_2px",
        ]:
            paper_rows.append({
                "stage": "stage4",
                "task": "localization",
                "subset": "class2_test_gold",
                "metric": metric_name,
                "value": class2_loc_metrics[metric_name],
                "role": "PAPER_CORE",
            })

    if class9_cls is not None:
        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]:
            paper_rows.append({
                "stage": "stage4",
                "task": "classification",
                "subset": "class9_inference2_gold",
                "metric": metric_name,
                "value": class9_cls[metric_name],
                "role": "PAPER_CORE",
            })

    pd.DataFrame(paper_rows).to_csv(
        PAPER_CORE_CSV,
        index=False,
    )

    classification_output = {
        "stage": "stage4_physics_multitask",
        "model_name": MODEL_NAME,
        "gold_csv": str(GOLD_CSV),
        "physics_config_file": str(PHYSICS_CONFIG_JSON),
        "fusion_alpha_file": str(FUSION_ALPHA_CSV),
        "combined_non_ambiguous": combined_cls,
        "class9_inference2_heldout": class9_cls,
        "class2_test_descriptive": class2_desc,
        "note": (
            "manual_label=-1 is excluded from classification metrics. "
            "The class9_inference2 subset is the stronger held-out "
            "classification generalization result."
        ),
    }

    with open(
        CLASSIFICATION_METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            classification_output,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    save_confusion_csv(
        combined_cls,
        CONFUSION_COMBINED_CSV,
    )

    save_confusion_csv(
        class9_cls,
        CONFUSION_CLASS9_CSV,
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("STAGE-4 COMBINED GOLD LOCALIZATION RESULTS")
    logger.info("=" * 78)

    logger.info(
        "Primary images             : %d",
        combined_loc_metrics["num_images"],
    )
    logger.info(
        "Total GT excitations       : %d",
        combined_loc_metrics["total_gt_excitations"],
    )
    logger.info(
        "Precision @ 2 px           : %.6f",
        combined_loc_metrics["precision_at_2px"],
    )
    logger.info(
        "Recall @ 2 px              : %.6f",
        combined_loc_metrics["recall_at_2px"],
    )
    logger.info(
        "F1 @ 2 px                  : %.6f",
        combined_loc_metrics["f1_at_2px"],
    )
    logger.info(
        "GT Match Rate @ 1 px       : %.6f",
        combined_loc_metrics["gt_match_rate_at_1px"],
    )
    logger.info(
        "GT Match Rate @ 2 px       : %.6f",
        combined_loc_metrics["gt_match_rate_at_2px"],
    )
    logger.info(
        "GT Match Rate @ 3 px       : %.6f",
        combined_loc_metrics["gt_match_rate_at_3px"],
    )
    logger.info(
        "Matched MAE @ 2 px         : %.6f px",
        combined_loc_metrics["matched_mae_at_2px"],
    )
    logger.info(
        "Exact Count Accuracy       : %.6f",
        combined_loc_metrics["exact_count_accuracy"],
    )
    logger.info(
        "Count MAE                  : %.6f",
        combined_loc_metrics["count_mae"],
    )
    logger.info(
        "Exact Sample Success @ 2px : %.6f",
        combined_loc_metrics["exact_sample_success_at_2px"],
    )
    logger.info(
        "False positives / image    : %.6f",
        combined_loc_metrics["false_positives_per_image"],
    )
    logger.info(
        "False negatives / image    : %.6f",
        combined_loc_metrics["false_negatives_per_image"],
    )

    logger.info("-" * 78)
    for _, row in per_source_df.iterrows():
        logger.info(
            "%s | N=%d | F1@2=%.6f | Recall@2=%.6f | "
            "ExactCount=%.6f | ExactSample=%.6f",
            row["source_group"],
            int(row["num_images"]),
            float(row["f1_at_2px"]),
            float(row["recall_at_2px"]),
            float(row["exact_count_accuracy"]),
            float(row["exact_sample_success_at_2px"]),
        )

    logger.info("=" * 78)
    logger.info("STAGE-4 GOLD CLASSIFICATION RESULTS")
    logger.info("=" * 78)

    logger.info(
        "Combined N=%d | Acc=%.6f | BalAcc=%.6f | MacroF1=%.6f",
        combined_cls["num_images"],
        combined_cls["accuracy"],
        combined_cls["balanced_accuracy"],
        combined_cls["macro_f1"],
    )

    if class9_cls is not None:
        logger.info(
            "Class9 held-out N=%d | Acc=%.6f | BalAcc=%.6f | MacroF1=%.6f",
            class9_cls["num_images"],
            class9_cls["accuracy"],
            class9_cls["balanced_accuracy"],
            class9_cls["macro_f1"],
        )

    if class2_desc is not None:
        logger.info(
            "Class2 test descriptive N=%d | class2 acc=%.6f | mean P2=%.6f",
            class2_desc["num_images"],
            class2_desc["class2_accuracy"],
            class2_desc["mean_prob_class2"],
        )

    # -----------------------------------------------------------------
    # Paper-oriented separated full reports
    # -----------------------------------------------------------------
    log_full_localization_metrics(
        logger=logger,
        title="STAGE-4 CLASS-2 GOLD LOCALIZATION — PAPER PRIMARY",
        metrics=class2_loc_metrics,
        paper_primary=True,
    )

    log_full_localization_metrics(
        logger=logger,
        title="STAGE-4 CLASS-9 GOLD LOCALIZATION — AUXILIARY",
        metrics=class9_loc_metrics,
        paper_primary=False,
    )

    log_full_classification_metrics(
        logger=logger,
        title="STAGE-4 CLASS-9 GOLD CLASSIFICATION — PAPER PRIMARY",
        metrics=class9_cls,
        paper_primary=True,
    )

    if class2_desc is not None:
        logger.info("=" * 78)
        logger.info("STAGE-4 CLASS-2 GOLD CLASSIFICATION — DESCRIPTIVE ONLY")
        logger.info("=" * 78)
        logger.info("[AUX] N                       : %d", class2_desc["num_images"])
        logger.info("[AUX] Class-2 Accuracy        : %.6f", class2_desc["class2_accuracy"])
        logger.info("[AUX] Mean P(class-0)         : %.6f", class2_desc["mean_prob_class0"])
        logger.info("[AUX] Mean P(class-1)         : %.6f", class2_desc["mean_prob_class1"])
        logger.info("[AUX] Mean P(class-2)         : %.6f", class2_desc["mean_prob_class2"])
        logger.info("[AUX] Predicted counts 0/1/2 : %s", class2_desc["predicted_class_counts"])

    logger.info("=" * 78)
    logger.info("PAPER-CORE OUTPUTS")
    logger.info("=" * 78)
    logger.info("Core JSON                    : %s", PAPER_CORE_JSON)
    logger.info("Core CSV                     : %s", PAPER_CORE_CSV)
    logger.info("Class2 localization JSON     : %s", CLASS2_LOCALIZATION_JSON)
    logger.info("Class9 classification JSON   : %s", CLASS9_CLASSIFICATION_JSON)
    logger.info("Physics config               : %s", PHYSICS_CONFIG_JSON)
    logger.info("Learned fusion alpha         : %s", FUSION_ALPHA_CSV)
    logger.info("Output directory             : %s", OUTPUT_DIR)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
