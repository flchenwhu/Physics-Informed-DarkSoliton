# !/usr/bin/env python3
# -*- coding: utf-8 -*-
###########flchen 2026.11 ###############
"""
train_multitask_baseline.py
===========================

Stage 3：全监督 Classification + Localization Multi-task Baseline。
"""

from pathlib import Path
import json
import logging
import random
import sys
import time

# import matplotlib
# matplotlib.use("Agg")  # 补充与进程之间的冲突  flchen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
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
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
NORMALIZATION_SOURCE_CSV = TRAIN_CSV

# MODEL_NAME = "simple_cnn"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
MODEL_NAME = "fasternet_t0"

NUM_CLASSES = 3
CLASS_NAMES = ["no_excitation", "single_excitation", "other_excitations"]

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 70   # 首次验证建议 3，确认后恢复 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

USE_CLASS_WEIGHTS = True

# Multi-task loss
LAMBDA_CLASSIFICATION = 1.0
LAMBDA_LOCALIZATION = 1.0
LAMBDA_OFFSET = 1.0

# Classification head
SIMPLE_CNN_CLASSIFIER_DROPOUT = 0.20
OTHER_CLASSIFIER_DROPOUT = 0.00

GRAD_CLIP_NORM = 5.0
USE_AMP = True

LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6

EARLY_STOP_PATIENCE = 12
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

OUTPUT_ROOT = ROOT / "output" / MODEL_NAME / "multitask_baseline"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoint"
LOG_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

BEST_CKPT = CHECKPOINT_DIR / "best.pt.tar"
LAST_CKPT = CHECKPOINT_DIR / "last.pt.tar"
LOG_FILE = LOG_DIR / "trainer.log"

# 每个 training batch 的原始 loss
BATCH_LOSS_CSV = METRICS_DIR / "batch_loss_history.csv"

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


# =============================================================================
# 3. Model
# =============================================================================

class ClassificationHead(nn.Module):
    """Deepest shared feature map -> global feature -> 3-class logits."""

    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, deepest_feature):
        feature = self.pool(deepest_feature)
        feature = torch.flatten(feature, 1)
        logits = self.fc(self.dropout(feature))
        return logits, feature


class DarkSolitonMultiTaskModel(nn.Module):
    """
    Shared backbone + classification head + Stage-2-compatible localization head.
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

    def forward(self, images):
        features = self.backbone(images)

        classification_logits, shared_feature = self.classification_head(
            features[-1]
        )
        localization = self.localization_head(features)

        return {
            "classification_logits": classification_logits,
            "shared_feature": shared_feature,
            "heatmap_logits": localization["heatmap_logits"],
            "offset_logits": localization["offset_logits"],
        }


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

def compute_multitask_loss(
    outputs,
    batch,
    classification_criterion,
    heatmap_criterion,
    offset_criterion,
):
    targets = build_multitask_localization_targets(batch)
    labels = targets["labels"]

    # Classification: all class-0/1/2
    loss_cls = classification_criterion(
        outputs["classification_logits"], labels
    )

    # Heatmap: only class-0/1; class-2 is NOT background
    loc_valid = targets["localization_valid"]
    if loc_valid.any():
        loss_heatmap = heatmap_criterion(
            outputs["heatmap_logits"][loc_valid],
            targets["heatmap_target"][loc_valid],
        )
    else:
        loss_heatmap = outputs["heatmap_logits"].sum() * 0.0

    # Offset: only class-1 GT center bin
    offset_mask = targets["offset_mask"]
    if offset_mask.any():
        pred_offsets = torch.sigmoid(
            outputs["offset_logits"][offset_mask]
        )
        gt_offsets = targets["offset_target"][offset_mask]
        loss_offset = offset_criterion(pred_offsets, gt_offsets)
    else:
        loss_offset = outputs["offset_logits"].sum() * 0.0

    loss_loc = loss_heatmap + LAMBDA_OFFSET * loss_offset

    loss_total = (
        LAMBDA_CLASSIFICATION * loss_cls
        + LAMBDA_LOCALIZATION * loss_loc
    )

    return {
        "loss": loss_total,
        "loss_classification": loss_cls,
        "loss_localization": loss_loc,
        "loss_heatmap": loss_heatmap,
        "loss_offset": loss_offset,
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
):
    model.train()

    keys = [
        "loss",
        "loss_classification",
        "loss_localization",
        "loss_heatmap",
        "loss_offset",
    ]
    sums = {k: 0.0 for k in keys}
    total_samples = 0
    batch_records = []

    for batch_index, batch in enumerate(loader, start=1):
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

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )

        scaler.step(optimizer)
        scaler.update()

        bs = images.shape[0]
        for k in keys:
            sums[k] += losses[k].item() * bs
        total_samples += bs

        batch_records.append({
            "epoch": int(epoch),
            "batch": int(batch_index),
            "global_step": int(
                (epoch - 1) * len(loader) + batch_index
            ),
            "batch_size": int(bs),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": float(
                losses["loss"].detach().item()
            ),
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
        })

    result = {
        k: v / total_samples
        for k, v in sums.items()
    }
    result["batch_records"] = batch_records
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
    ]
    sums = {k: 0.0 for k in keys}
    total_samples = 0

    y_true, y_pred = [], []
    probability_rows, file_names = [], []
    feature_rows = []
    localization_cache = []

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

        # Strict localization metrics only for class-0/1
        for i in range(bs):
            label = int(labels[i])
            if label not in {0, 1}:
                continue

            localization_cache.append({
                "file_name": batch["file_name"][i],
                "label": label,
                "gt_position": float(positions[i]) if label == 1 else None,
                "heatmap_probabilities": hm_probs[i].astype(np.float32),
                "offset_probabilities": off_probs[i].astype(np.float32),
            })

    cls_metrics = compute_classification_metrics(
        np.asarray(y_true), np.asarray(y_pred)
    )

    loc_metrics, loc_records = localization_metrics_from_cache(
        localization_cache, localization_threshold
    )

    joint_score = harmonic_joint_score(
        cls_metrics["macro_f1"],
        loc_metrics["f1_at_2px"],
    )

    return {
        **{k: v / total_samples for k, v in sums.items()},
        "classification_metrics": cls_metrics,
        "localization_metrics": loc_metrics,
        "joint_score": joint_score,
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "probabilities": np.asarray(probability_rows, dtype=np.float32),
        "file_names": file_names,
        "localization_cache": localization_cache,
        "localization_records": loc_records,
        "features": (
            np.asarray(feature_rows, dtype=np.float32)
            if collect_features
            else None
        ),
    }


# =============================================================================
# 9. Validation threshold search — cached, no repeated inference
# =============================================================================

def search_best_localization_threshold(localization_cache):
    rows = []
    best_threshold = None
    best_metrics = None
    best_f1 = -1.0

    for threshold in THRESHOLD_SEARCH_VALUES:
        metrics, _ = localization_metrics_from_cache(
            localization_cache, float(threshold)
        )

        rows.append({
            "threshold": float(threshold),
            "precision_at_2px": metrics["precision_at_2px"],
            "recall_at_2px": metrics["recall_at_2px"],
            "f1_at_2px": metrics["f1_at_2px"],
            "acc_at_1px": metrics["acc_at_1px"],
            "acc_at_2px": metrics["acc_at_2px"],
            "acc_at_3px": metrics["acc_at_3px"],
            "top1_mae_px": metrics["top1_mae_px"],
            "matched_mae_px": metrics["matched_mae_px"],
            "false_positives_per_image": (
                metrics["false_positives_per_image"]
            ),
        })

        current_f1 = metrics["f1_at_2px"]
        better = current_f1 > best_f1 + 1e-12

        if (
            not better
            and best_metrics is not None
            and abs(current_f1 - best_f1) <= 1e-12
        ):
            current_precision = metrics["precision_at_2px"]
            best_precision = best_metrics["precision_at_2px"]

            if current_precision > best_precision + 1e-12:
                better = True
            elif abs(current_precision - best_precision) <= 1e-12:
                current_fp = metrics["false_positives_per_image"]
                best_fp = best_metrics["false_positives_per_image"]

                if current_fp < best_fp - 1e-12:
                    better = True
                elif (
                    abs(current_fp - best_fp) <= 1e-12
                    and float(threshold) > float(best_threshold)
                ):
                    better = True

        if better:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = float(current_f1)

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(
        METRICS_DIR / "val_threshold_search.csv",
        index=False,
    )

    return best_threshold, best_metrics, threshold_df


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
):
    payload = {
        "epoch": int(epoch),
        "stage": "stage3_multitask",
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_joint_score": float(best_joint_score),
        "best_epoch": int(best_epoch),
        "no_improve_count": int(no_improve_count),
        "history": history,
        "normalization": {
            "mean": float(normalization_mean),
            "std": float(normalization_std),
        },
        "class_weights": class_weights.detach().cpu().tolist(),
        "loss_weights": {
            "lambda_classification": LAMBDA_CLASSIFICATION,
            "lambda_localization": LAMBDA_LOCALIZATION,
            "lambda_offset": LAMBDA_OFFSET,
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
):
    checkpoint = torch.load(path_, map_location=DEVICE)

    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("Checkpoint MODEL_NAME mismatch.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

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
        "stage": "Stage 3",
        "task": "supervised_classification_localization_multitask",
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
            "class2_localization_masked": True,
        },
        "loss": {
            "formula": (
                "lambda_cls*L_cls + "
                "lambda_loc*(L_heatmap + lambda_offset*L_offset)"
            ),
            "lambda_classification": LAMBDA_CLASSIFICATION,
            "lambda_localization": LAMBDA_LOCALIZATION,
            "lambda_offset": LAMBDA_OFFSET,
            "classification_loss": "WeightedCrossEntropyLoss",
            "heatmap_loss": "CenterNet-style ModifiedFocalLoss",
            "offset_loss": "SmoothL1Loss",
        },
        "checkpoint_selection": {
            "classification_metric": "val Macro-F1",
            "localization_metric": "val F1@2px",
            "localization_threshold": VAL_CHECKPOINT_THRESHOLD,
            "joint_score": (
                "harmonic_mean(classification Macro-F1, "
                "localization F1@2px)"
            ),
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
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train total",
        linewidth=1.5,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation total",
        linewidth=1.5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Multi-task loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(
        FIGURE_DIR / "loss_curves.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Validation task scores
    fig = plt.figure(figsize=(6.4, 4.6))
    plt.plot(
        history_df["epoch"],
        history_df["val_classification_macro_f1"],
        label="Classification Macro-F1",
        linewidth=1.5,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_localization_f1_at_2px"],
        label="Localization F1@2px",
        linewidth=1.5,
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_joint_score"],
        label="Joint score",
        linewidth=1.5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(
        FIGURE_DIR / "val_task_scores.png",
        dpi=600,
        bbox_inches="tight",
    )
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
        "threshold_selected_from": "validation only",
        "validation_localization_at_selected_threshold": (
            selected_val_metrics
        ),
        "test_loss": float(test_result["loss"]),
        "test_loss_classification": float(
            test_result["loss_classification"]
        ),
        "test_loss_localization": float(
            test_result["loss_localization"]
        ),
        "test_loss_heatmap": float(test_result["loss_heatmap"]),
        "test_loss_offset": float(test_result["loss_offset"]),
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
    prepare_output_dirs()

    if not RESUME and BATCH_LOSS_CSV.exists():
        BATCH_LOSS_CSV.unlink()

    logger = build_logger()
    set_seed(SEED)

    logger.info("=" * 78)
    logger.info(
        "Dark Soliton Stage 3 — "
        "Supervised Classification + Localization Multi-task"
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
        "Loss          : %.2f*L_cls + %.2f*(L_heatmap + %.2f*L_offset)",
        LAMBDA_CLASSIFICATION,
        LAMBDA_LOCALIZATION,
        LAMBDA_OFFSET,
    )
    logger.info(
        "Checkpoint    : harmonic mean of val Macro-F1 and val F1@2px"
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
    logger.info(
        "Test : N=%d, counts=%s",
        len(test_dataset),
        test_dataset.class_counts(),
    )

    # Model
    model = DarkSolitonMultiTaskModel(
        MODEL_NAME, NUM_CLASSES
    ).to(DEVICE)

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

    heatmap_criterion = detector_base.ModifiedFocalLoss(
        alpha=detector_base.FOCAL_ALPHA,
        beta=detector_base.FOCAL_BETA,
    )
    offset_criterion = nn.SmoothL1Loss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )
    logger.info("AMP enabled   : %s", amp_enabled)

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
        )

    # Training
    training_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()

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
        )

        pd.DataFrame(
            train_result["batch_records"]
        ).to_csv(
            BATCH_LOSS_CSV,
            mode="a",
            header=not BATCH_LOSS_CSV.exists(),
            index=False,
        )

        val_result = evaluate_model(
            model,
            val_loader,
            classification_criterion,
            heatmap_criterion,
            offset_criterion,
            VAL_CHECKPOINT_THRESHOLD,
            collect_features=False,
        )

        scheduler.step(val_result["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        val_cls = val_result["classification_metrics"]
        val_loc = val_result["localization_metrics"]
        joint = float(val_result["joint_score"])

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_result["loss"],
            "train_loss_classification": (
                train_result["loss_classification"]
            ),
            "train_loss_localization": (
                train_result["loss_localization"]
            ),
            "train_loss_heatmap": train_result["loss_heatmap"],
            "train_loss_offset": train_result["loss_offset"],

            "val_loss": val_result["loss"],
            "val_loss_classification": (
                val_result["loss_classification"]
            ),
            "val_loss_localization": (
                val_result["loss_localization"]
            ),
            "val_loss_heatmap": val_result["loss_heatmap"],
            "val_loss_offset": val_result["loss_offset"],

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

            "val_joint_score": joint,
            "epoch_seconds": time.time() - epoch_start,
        }
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
        )

        pd.DataFrame(history).to_csv(
            METRICS_DIR / "training_history.csv",
            index=False,
        )

        marker = " *BEST*" if improved else ""

        logger.info(
            "Epoch %03d/%03d | lr=%.2e | "
            "train=%.5f (cls=%.5f,loc=%.5f,hm=%.5f,off=%.5f) | "
            "val=%.5f | Cls MacroF1=%.4f | "
            "Loc P/R/F1@2=%.4f/%.4f/%.4f | Joint=%.4f | %.1fs%s",
            epoch,
            MAX_EPOCHS,
            current_lr,
            train_result["loss"],
            train_result["loss_classification"],
            train_result["loss_localization"],
            train_result["loss_heatmap"],
            train_result["loss_offset"],
            val_result["loss"],
            val_cls["macro_f1"],
            val_loc["precision_at_2px"],
            val_loc["recall_at_2px"],
            val_loc["f1_at_2px"],
            joint,
            row["epoch_seconds"],
            marker,
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
    save_training_figures(history_df)

    logger.info(
        "Training finished | %.2f min",
        (time.time() - training_start) / 60.0,
    )
    logger.info(
        "Best epoch=%d | best val JointScore=%.6f",
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

    selected_threshold, selected_val_loc, _ = (
        search_best_localization_threshold(
            best_val_result["localization_cache"]
        )
    )

    logger.info(
        "Selected validation localization threshold=%.2f | "
        "P@2=%.4f R@2=%.4f F1@2=%.4f",
        selected_threshold,
        selected_val_loc["precision_at_2px"],
        selected_val_loc["recall_at_2px"],
        selected_val_loc["f1_at_2px"],
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
        selected_val_loc,
    )

    test_cls = test_result["classification_metrics"]
    test_loc = test_result["localization_metrics"]

    logger.info("=" * 78)
    logger.info("FINAL TEST RESULTS — BEST MULTI-TASK CHECKPOINT")
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
