# !/usr/bin/env python3
# -*- coding: utf-8 -*-
###########flchen 2026.11 ###############
"""
train_detector_baseline.py
==========================

Dark Soliton 项目 Stage 2：全监督 Localization Baseline。

"""

from pathlib import Path
import json
import logging
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # 补充与进程之间的冲突  flchen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18

from dark_soliton_dataloader import (
    DarkSolitonDataset,
    make_dataloader,
    compute_dataset_mean_std,
)


# ======================================================================
# 1. Configuration
# ======================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "detection_train.csv"
VAL_CSV = DATA_DIR / "detection_val.csv"
TEST_CSV = DATA_DIR / "detection_test.csv"

# 为 Stage 1~4 保持统一输入尺度
NORMALIZATION_SOURCE_CSV = DATA_DIR / "train.csv"

# MODEL_NAME = "simple_cnn"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
MODEL_NAME = "fasternet_t0"

IMAGE_WIDTH = 164
NUM_BINS = 82
BIN_WIDTH = IMAGE_WIDTH / NUM_BINS

if abs(BIN_WIDTH - 2.0) > 1e-12:
    raise ValueError(f"Expected BIN_WIDTH=2.0, got {BIN_WIDTH}")

GAUSSIAN_SIGMA_BINS = 1.0
GAUSSIAN_RADIUS_BINS = 3

NMS_RADIUS_BINS = 1

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0
LAMBDA_OFFSET = 1.0

SEED = 2027        # 跑3轮随机种子   取mean+std  (42  90  2027)
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 70    # epoch的数量  100 或者50都可以
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
USE_AMP = True

FUSION_DIM = 64
HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT = 0.10

LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6

EARLY_STOP_PATIENCE = 12  # 早停 
MIN_DELTA = 1e-4

PRIMARY_TOLERANCE_PX = 2.0

# 用于每个 epoch 选择 best checkpoint
VAL_CHECKPOINT_THRESHOLD = 0.30

# best checkpoint 训练完成后，仅在 validation 上搜索
THRESHOLD_SEARCH_VALUES = np.round(
    np.arange(0.05, 0.91, 0.05),
    2,
)

RESUME = False
CHECK_FILE_EXISTS = True

OUTPUT_ROOT = (
    ROOT
    / "output"
    / MODEL_NAME
    / "detector_baseline"
)

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoint"
LOG_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

BEST_CKPT = CHECKPOINT_DIR / "best.pt.tar"
LAST_CKPT = CHECKPOINT_DIR / "last.pt.tar"
LOG_FILE = LOG_DIR / "trainer.log"

# 每个 training batch 的原始 loss
BATCH_LOSS_CSV = METRICS_DIR / "batch_loss_history.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# 2. General utilities
# ======================================================================

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
    for path_ in [
        OUTPUT_ROOT,
        CHECKPOINT_DIR,
        LOG_DIR,
        METRICS_DIR,
        FIGURE_DIR,
    ]:
        path_.mkdir(parents=True, exist_ok=True)


def build_logger():
    logger = logging.getLogger("dark_soliton_detector")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


# ======================================================================
# 3. Backbones
# ======================================================================

class SimpleCNNFeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [16, 32, 64, 128]

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class ResNet18FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        base = resnet18(weights=None)

        base.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "RepViT/FasterNet 需要 timm。\n"
            "建议运行：mamba install timm"
        ) from exc
    return timm


class TimmFeatureBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        timm = _import_timm()

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1,
            features_only=True,
        )

        self.out_channels = list(
            self.model.feature_info.channels()
        )

        if len(self.out_channels) < 2:
            raise RuntimeError(
                f"{model_name}: insufficient feature scales: "
                f"{self.out_channels}"
            )

    def forward(self, x):
        features = self.model(x)

        if not isinstance(features, (list, tuple)):
            raise TypeError(
                "features_only backbone must return list/tuple."
            )

        return list(features)


def build_feature_backbone(model_name):
    model_name = model_name.lower().strip()

    if model_name == "simple_cnn":
        return SimpleCNNFeatureBackbone()

    if model_name == "resnet18":
        return ResNet18FeatureBackbone()

    if model_name in {
        "repvit_m0_9",
        "fasternet_t0",
    }:
        return TimmFeatureBackbone(model_name)

    raise ValueError(
        f"Unsupported MODEL_NAME={model_name!r}"
    )


# ======================================================================
# 4. Localization head
# ======================================================================

class MultiScaleLocalizationHead(nn.Module):
    def __init__(
        self,
        in_channels_list,
        num_bins=82,
        fusion_dim=64,
        hidden_dim=128,
        dropout=0.10,
    ):
        super().__init__()

        self.num_bins = int(num_bins)

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    fusion_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )
            for in_channels in in_channels_list
        ])

        fused_channels = (
            fusion_dim
            * len(in_channels_list)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(
                fused_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.heatmap_head = nn.Conv1d(
            hidden_dim,
            1,
            kernel_size=1,
        )

        self.offset_head = nn.Conv1d(
            hidden_dim,
            1,
            kernel_size=1,
        )

        # 训练初期默认低 positive prior，减少全图伪峰。
        prior_probability = 0.01
        heatmap_bias = -np.log(
            (1.0 - prior_probability)
            / prior_probability
        )

        nn.init.constant_(
            self.heatmap_head.bias,
            float(heatmap_bias),
        )

        nn.init.zeros_(
            self.offset_head.bias
        )

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError(
                "Feature map count != projection count."
            )

        projected = []

        for feature, projection in zip(
            features,
            self.projections,
        ):
            x = projection(feature)

            # [B,C,H,W] -> [B,C,W]
            x = x.mean(dim=2)

            if x.shape[-1] != self.num_bins:
                x = F.interpolate(
                    x,
                    size=self.num_bins,
                    mode="linear",
                    align_corners=False,
                )

            projected.append(x)

        fused = torch.cat(projected, dim=1)
        fused = self.fusion(fused)

        heatmap_logits = self.heatmap_head(
            fused
        ).squeeze(1)

        offset_logits = self.offset_head(
            fused
        ).squeeze(1)

        return {
            "heatmap_logits": heatmap_logits,
            "offset_logits": offset_logits,
        }


class DarkSolitonDetector(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone = build_feature_backbone(
            model_name
        )

        self.head = MultiScaleLocalizationHead(
            in_channels_list=self.backbone.out_channels,
            num_bins=NUM_BINS,
            fusion_dim=FUSION_DIM,
            hidden_dim=HEAD_HIDDEN_DIM,
            dropout=HEAD_DROPOUT,
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features)



# ======================================================================
# 5. Data / normalization
# ======================================================================

def compute_project_normalization(logger):
    logger.info(
        "Computing normalization from TRAIN split only: %s",
        NORMALIZATION_SOURCE_CSV,
    )

    normalization_dataset = DarkSolitonDataset(
        csv_path=NORMALIZATION_SOURCE_CSV,
        task="classification",
        normalize="none",
        check_file_exists=CHECK_FILE_EXISTS,
    )

    mean, std = compute_dataset_mean_std(
        normalization_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    if std <= 0:
        raise RuntimeError(
            f"Invalid normalization std={std}"
        )

    payload = {
        "method": "dataset_zscore",
        "mean": float(mean),
        "std": float(std),
        "source": str(
            NORMALIZATION_SOURCE_CSV
        ),
        "note": (
            "Computed from supervised train.csv only. "
            "Validation/test were not used."
        ),
    }

    with open(
        METRICS_DIR / "normalization.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Normalization: mean=%.8f, std=%.8f",
        mean,
        std,
    )

    return float(mean), float(std)


def build_datasets_and_loaders(
    mean,
    std,
):
    common_kwargs = dict(
        task="detection",
        normalize="dataset_zscore",
        dataset_mean=mean,
        dataset_std=std,
        check_file_exists=CHECK_FILE_EXISTS,
    )

    train_dataset = DarkSolitonDataset(
        csv_path=TRAIN_CSV,
        **common_kwargs,
    )

    val_dataset = DarkSolitonDataset(
        csv_path=VAL_CSV,
        **common_kwargs,
    )

    test_dataset = DarkSolitonDataset(
        csv_path=TEST_CSV,
        **common_kwargs,
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_loader = make_dataloader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# ======================================================================
# 6. 82-bin Gaussian targets
# ======================================================================

def build_heatmap_targets(batch):
    """
    从连续 position_x 构造 82-bin Gaussian center target。

    不使用旧 41-cell CSV 中的 cell_index / cell_offset。
    """
    positions = batch["position"].to(
        DEVICE,
        non_blocking=(DEVICE.type == "cuda"),
    ).float()

    position_valid = batch[
        "position_valid"
    ].to(
        DEVICE,
        non_blocking=(DEVICE.type == "cuda"),
    ).bool()

    batch_size = positions.shape[0]

    heatmap_target = torch.zeros(
        batch_size,
        NUM_BINS,
        dtype=torch.float32,
        device=DEVICE,
    )

    offset_target = torch.zeros(
        batch_size,
        NUM_BINS,
        dtype=torch.float32,
        device=DEVICE,
    )

    offset_mask = torch.zeros(
        batch_size,
        NUM_BINS,
        dtype=torch.bool,
        device=DEVICE,
    )

    positive_rows = torch.nonzero(
        position_valid,
        as_tuple=False,
    ).squeeze(1)

    if positive_rows.numel() == 0:
        return (
            heatmap_target,
            offset_target,
            offset_mask,
        )

    positive_positions = positions[
        position_valid
    ]

    if (
        (positive_positions < 0.0).any()
        or (
            positive_positions
            > (IMAGE_WIDTH - 1)
        ).any()
    ):
        raise ValueError(
            "GT position outside valid pixel range [0,163]."
        )

    scaled_positions = (
        positive_positions
        / BIN_WIDTH
    )

    center_bins = torch.floor(
        scaled_positions
    ).long()

    center_bins = torch.clamp(
        center_bins,
        0,
        NUM_BINS - 1,
    )

    offsets = (
        scaled_positions
        - center_bins.float()
    )

    offsets = torch.clamp(
        offsets,
        0.0,
        1.0 - 1e-7,
    )

    for row_index, center_bin, offset in zip(
        positive_rows.tolist(),
        center_bins.tolist(),
        offsets.tolist(),
    ):
        left = max(
            0,
            center_bin - GAUSSIAN_RADIUS_BINS,
        )

        right = min(
            NUM_BINS - 1,
            center_bin + GAUSSIAN_RADIUS_BINS,
        )

        bin_indices = torch.arange(
            left,
            right + 1,
            device=DEVICE,
            dtype=torch.float32,
        )

        distances = (
            bin_indices
            - float(center_bin)
        )

        gaussian_values = torch.exp(
            -0.5
            * (
                distances
                / GAUSSIAN_SIGMA_BINS
            ) ** 2
        )

        heatmap_target[
            row_index,
            left:right + 1,
        ] = torch.maximum(
            heatmap_target[
                row_index,
                left:right + 1,
            ],
            gaussian_values,
        )

        # 中心峰严格置为 1
        heatmap_target[
            row_index,
            center_bin,
        ] = 1.0

        offset_target[
            row_index,
            center_bin,
        ] = float(offset)

        offset_mask[
            row_index,
            center_bin,
        ] = True

    return (
        heatmap_target,
        offset_target,
        offset_mask,
    )


# ======================================================================
# 7. Loss
# ======================================================================

class ModifiedFocalLoss(nn.Module):
    """
    CenterNet-style modified focal loss。

    target == 1:
        positive center

    target < 1:
        negative / soft near-center region

    Gaussian 邻域通过：
        (1-target)^beta
    降低负样本权重。
    """

    def __init__(
        self,
        alpha=2.0,
        beta=4.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(
        self,
        logits,
        target,
    ):
        pred = torch.sigmoid(logits)

        pred = torch.clamp(
            pred,
            min=1e-6,
            max=1.0 - 1e-6,
        )

        positive_mask = (
            target >= 1.0 - 1e-6
        ).float()

        negative_mask = (
            target < 1.0 - 1e-6
        ).float()

        negative_weights = (
            (1.0 - target)
            ** self.beta
        )

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

        num_positive = positive_mask.sum()

        if num_positive > 0:
            return (
                positive_loss.sum()
                + negative_loss.sum()
            ) / num_positive

        return negative_loss.sum()


def compute_detection_loss(
    outputs,
    batch,
    heatmap_criterion,
    offset_criterion,
):
    (
        heatmap_target,
        offset_target,
        offset_mask,
    ) = build_heatmap_targets(batch)

    heatmap_logits = outputs[
        "heatmap_logits"
    ]

    offset_logits = outputs[
        "offset_logits"
    ]

    loss_heatmap = heatmap_criterion(
        heatmap_logits,
        heatmap_target,
    )

    if offset_mask.any():
        predicted_offsets = torch.sigmoid(
            offset_logits[offset_mask]
        )

        gt_offsets = offset_target[
            offset_mask
        ]

        loss_offset = offset_criterion(
            predicted_offsets,
            gt_offsets,
        )
    else:
        loss_offset = (
            heatmap_logits.sum()
            * 0.0
        )

    loss_total = (
        loss_heatmap
        + LAMBDA_OFFSET
        * loss_offset
    )

    return (
        loss_total,
        loss_heatmap,
        loss_offset,
    )


# ======================================================================
# 8. Decoding
# ======================================================================

def find_local_peak_bins(
    probabilities,
    threshold,
):
    """
    局部极大值 + 1D greedy NMS。
    """
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    local_peaks = []

    for index in range(
        len(probabilities)
    ):
        value = probabilities[index]

        if value < threshold:
            continue

        left_value = (
            probabilities[index - 1]
            if index > 0
            else -np.inf
        )

        right_value = (
            probabilities[index + 1]
            if index < len(probabilities) - 1
            else -np.inf
        )

        if (
            value >= left_value
            and value >= right_value
        ):
            local_peaks.append(index)

    if not local_peaks:
        return []

    ranked = sorted(
        local_peaks,
        key=lambda idx: probabilities[idx],
        reverse=True,
    )

    selected = []

    for candidate in ranked:
        too_close = any(
            abs(
                candidate
                - selected_bin
            ) <= NMS_RADIUS_BINS
            for selected_bin in selected
        )

        if not too_close:
            selected.append(candidate)

    return sorted(selected)


def continuous_position_from_bin(
    bin_index,
    offset_probability,
):
    position = (
        float(bin_index)
        * BIN_WIDTH
        + float(offset_probability)
        * BIN_WIDTH
    )

    return float(
        np.clip(
            position,
            0.0,
            IMAGE_WIDTH - 1.0,
        )
    )


def decode_single_prediction(
    heatmap_logits,
    offset_logits,
    threshold,
):
    heatmap_probabilities = (
        torch.sigmoid(
            heatmap_logits
        )
        .detach()
        .cpu()
        .numpy()
    )

    offset_probabilities = (
        torch.sigmoid(
            offset_logits
        )
        .detach()
        .cpu()
        .numpy()
    )

    peak_bins = find_local_peak_bins(
        probabilities=heatmap_probabilities,
        threshold=threshold,
    )

    pred_positions = []
    pred_scores = []

    for bin_index in peak_bins:
        pred_positions.append(
            continuous_position_from_bin(
                bin_index=bin_index,
                offset_probability=offset_probabilities[
                    bin_index
                ],
            )
        )

        pred_scores.append(
            float(
                heatmap_probabilities[
                    bin_index
                ]
            )
        )

    # Top-1 不依赖 threshold
    top1_bin = int(
        np.argmax(
            heatmap_probabilities
        )
    )

    top1_position = continuous_position_from_bin(
        bin_index=top1_bin,
        offset_probability=offset_probabilities[
            top1_bin
        ],
    )

    top1_score = float(
        heatmap_probabilities[
            top1_bin
        ]
    )

    return {
        "pred_positions": pred_positions,
        "pred_scores": pred_scores,
        "top1_position": top1_position,
        "top1_score": top1_score,
    }


# ======================================================================
# 9. Metrics
# ======================================================================

def evaluate_prediction_records(
    records,
    primary_tolerance_px=2.0,
):
    tp = 0
    fp = 0
    fn = 0

    matched_errors = []
    top1_errors = []

    positive_total = 0

    hits = {
        1.0: 0,
        2.0: 0,
        3.0: 0,
    }

    total_predictions = 0

    for record in records:
        gt_position = record[
            "gt_position"
        ]

        pred_positions = record[
            "pred_positions"
        ]

        total_predictions += len(
            pred_positions
        )

        # class-0：thresholded predictions 全部为 FP
        if gt_position is None:
            fp += len(
                pred_positions
            )
            continue

        # class-1
        positive_total += 1

        top1_error = abs(
            float(
                record[
                    "top1_position"
                ]
            )
            - float(
                gt_position
            )
        )

        top1_errors.append(
            top1_error
        )

        if len(pred_positions) == 0:
            fn += 1
            continue

        distances = np.abs(
            np.asarray(
                pred_positions,
                dtype=np.float64,
            )
            - float(
                gt_position
            )
        )

        nearest_error = float(
            distances.min()
        )

        for tolerance in hits:
            if nearest_error <= tolerance:
                hits[
                    tolerance
                ] += 1

        if (
            nearest_error
            <= primary_tolerance_px
        ):
            tp += 1

            # 最近的一个与 GT 匹配，剩余候选都是 FP
            fp += (
                len(pred_positions)
                - 1
            )

            matched_errors.append(
                nearest_error
            )
        else:
            fn += 1

            # 一个都没有命中
            fp += len(
                pred_positions
            )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    matched_mae = (
        float(
            np.mean(
                matched_errors
            )
        )
        if matched_errors
        else float("nan")
    )

    top1_mae = (
        float(
            np.mean(
                top1_errors
            )
        )
        if top1_errors
        else float("nan")
    )

    acc_at_1 = (
        hits[1.0]
        / positive_total
        if positive_total > 0
        else 0.0
    )

    acc_at_2 = (
        hits[2.0]
        / positive_total
        if positive_total > 0
        else 0.0
    )

    acc_at_3 = (
        hits[3.0]
        / positive_total
        if positive_total > 0
        else 0.0
    )

    false_positives_per_image = (
        fp / len(records)
        if records
        else 0.0
    )

    mean_predictions_per_image = (
        total_predictions
        / len(records)
        if records
        else 0.0
    )

    return {
        "precision_at_2px": float(
            precision
        ),
        "recall_at_2px": float(
            recall
        ),
        "f1_at_2px": float(
            f1
        ),
        "acc_at_1px": float(
            acc_at_1
        ),
        "acc_at_2px": float(
            acc_at_2
        ),
        "acc_at_3px": float(
            acc_at_3
        ),
        "top1_mae_px": float(
            top1_mae
        ),
        "matched_mae_px": float(
            matched_mae
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "positive_samples": int(
            positive_total
        ),
        "false_positives_per_image": float(
            false_positives_per_image
        ),
        "mean_predictions_per_image": float(
            mean_predictions_per_image
        ),
    }



# ======================================================================
# 10. Train / evaluate
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    amp_enabled,
    heatmap_criterion,
    offset_criterion,
    epoch,
):
    model.train()

    sum_total = 0.0
    sum_heatmap = 0.0
    sum_offset = 0.0
    total_samples = 0
    batch_records = []

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)

            (
                loss_total,
                loss_heatmap,
                loss_offset,
            ) = compute_detection_loss(
                outputs=outputs,
                batch=batch,
                heatmap_criterion=heatmap_criterion,
                offset_criterion=offset_criterion,
            )

        scaler.scale(
            loss_total
        ).backward()

        scaler.unscale_(
            optimizer
        )

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]

        sum_total += (
            loss_total.item()
            * batch_size
        )

        sum_heatmap += (
            loss_heatmap.item()
            * batch_size
        )

        sum_offset += (
            loss_offset.item()
            * batch_size
        )

        total_samples += batch_size

        batch_records.append({
            "epoch": int(epoch),
            "batch": int(batch_index),
            "global_step": int(
                (epoch - 1) * len(loader) + batch_index
            ),
            "batch_size": int(batch_size),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": float(loss_total.detach().item()),
            "loss_heatmap": float(
                loss_heatmap.detach().item()
            ),
            "loss_offset": float(
                loss_offset.detach().item()
            ),
        })

    return {
        "loss": (
            sum_total
            / total_samples
        ),
        "loss_heatmap": (
            sum_heatmap
            / total_samples
        ),
        "loss_offset": (
            sum_offset
            / total_samples
        ),
        "batch_records": batch_records,
    }


@torch.no_grad()
def evaluate_detector(
    model,
    loader,
    heatmap_criterion,
    offset_criterion,
    threshold,
):
    model.eval()

    sum_total = 0.0
    sum_heatmap = 0.0
    sum_offset = 0.0
    total_samples = 0

    records = []

    for batch in loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        outputs = model(images)

        (
            loss_total,
            loss_heatmap,
            loss_offset,
        ) = compute_detection_loss(
            outputs=outputs,
            batch=batch,
            heatmap_criterion=heatmap_criterion,
            offset_criterion=offset_criterion,
        )

        batch_size = images.shape[0]

        sum_total += (
            loss_total.item()
            * batch_size
        )

        sum_heatmap += (
            loss_heatmap.item()
            * batch_size
        )

        sum_offset += (
            loss_offset.item()
            * batch_size
        )

        total_samples += batch_size

        labels = (
            batch["label"]
            .cpu()
            .numpy()
            .astype(int)
        )

        gt_positions = (
            batch["position"]
            .cpu()
            .numpy()
            .astype(float)
        )

        heatmap_logits = outputs[
            "heatmap_logits"
        ]

        offset_logits = outputs[
            "offset_logits"
        ]

        for i in range(batch_size):
            decoded = decode_single_prediction(
                heatmap_logits=heatmap_logits[i],
                offset_logits=offset_logits[i],
                threshold=threshold,
            )

            gt_position = (
                float(
                    gt_positions[i]
                )
                if labels[i] == 1
                else None
            )

            records.append({
                "file_name": batch[
                    "file_name"
                ][i],
                "label": int(
                    labels[i]
                ),
                "gt_position": gt_position,
                "pred_positions": decoded[
                    "pred_positions"
                ],
                "pred_scores": decoded[
                    "pred_scores"
                ],
                "top1_position": decoded[
                    "top1_position"
                ],
                "top1_score": decoded[
                    "top1_score"
                ],
            })

    metrics = evaluate_prediction_records(
        records,
        primary_tolerance_px=PRIMARY_TOLERANCE_PX,
    )

    return {
        "loss": (
            sum_total
            / total_samples
        ),
        "loss_heatmap": (
            sum_heatmap
            / total_samples
        ),
        "loss_offset": (
            sum_offset
            / total_samples
        ),
        "metrics": metrics,
        "records": records,
    }


# ======================================================================
# 11. Validation threshold search
# ======================================================================

@torch.no_grad()
def search_best_threshold(
    model,
    val_loader,
    heatmap_criterion,
    offset_criterion,
):
    rows = []

    best_threshold = None
    best_f1 = -1.0
    best_metrics = None

    for threshold in (
        THRESHOLD_SEARCH_VALUES
    ):
        result = evaluate_detector(
            model=model,
            loader=val_loader,
            heatmap_criterion=heatmap_criterion,
            offset_criterion=offset_criterion,
            threshold=float(
                threshold
            ),
        )

        metrics = result[
            "metrics"
        ]

        rows.append({
            "threshold": float(
                threshold
            ),
            "precision_at_2px": (
                metrics[
                    "precision_at_2px"
                ]
            ),
            "recall_at_2px": (
                metrics[
                    "recall_at_2px"
                ]
            ),
            "f1_at_2px": (
                metrics[
                    "f1_at_2px"
                ]
            ),
            "acc_at_1px": (
                metrics[
                    "acc_at_1px"
                ]
            ),
            "acc_at_2px": (
                metrics[
                    "acc_at_2px"
                ]
            ),
            "acc_at_3px": (
                metrics[
                    "acc_at_3px"
                ]
            ),
            "top1_mae_px": (
                metrics[
                    "top1_mae_px"
                ]
            ),
            "matched_mae_px": (
                metrics[
                    "matched_mae_px"
                ]
            ),
            "false_positives_per_image": (
                metrics[
                    "false_positives_per_image"
                ]
            ),
        })

        current_f1 = metrics[
            "f1_at_2px"
        ]

        better = (
            current_f1
            > best_f1
            + 1e-12
        )

        if (
            not better
            and best_metrics is not None
            and abs(
                current_f1
                - best_f1
            ) <= 1e-12
        ):
            current_precision = metrics[
                "precision_at_2px"
            ]

            best_precision = best_metrics[
                "precision_at_2px"
            ]

            if (
                current_precision
                > best_precision
                + 1e-12
            ):
                better = True

            elif abs(
                current_precision
                - best_precision
            ) <= 1e-12:
                current_fp = metrics[
                    "false_positives_per_image"
                ]

                best_fp = best_metrics[
                    "false_positives_per_image"
                ]

                if (
                    current_fp
                    < best_fp
                    - 1e-12
                ):
                    better = True

                elif (
                    abs(
                        current_fp
                        - best_fp
                    ) <= 1e-12
                    and float(
                        threshold
                    )
                    > float(
                        best_threshold
                    )
                ):
                    better = True

        if better:
            best_threshold = float(
                threshold
            )

            best_f1 = float(
                current_f1
            )

            best_metrics = metrics

    threshold_df = pd.DataFrame(
        rows
    )

    threshold_df.to_csv(
        METRICS_DIR
        / "val_threshold_search.csv",
        index=False,
    )

    return (
        best_threshold,
        best_metrics,
        threshold_df,
    )


# ======================================================================
# 12. Checkpoints
# ======================================================================

def save_checkpoint(
    path_,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_f1,
    best_epoch,
    no_improve_count,
    history,
    normalization_mean,
    normalization_std,
):
    payload = {
        "epoch": int(epoch),
        "model_name": MODEL_NAME,

        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "scaler_state_dict": (
            scaler.state_dict()
        ),

        "best_val_f1_at_2px": float(
            best_val_f1
        ),
        "best_epoch": int(
            best_epoch
        ),
        "no_improve_count": int(
            no_improve_count
        ),

        "history": history,

        "normalization": {
            "mean": float(
                normalization_mean
            ),
            "std": float(
                normalization_std
            ),
        },

        "image_width": IMAGE_WIDTH,
        "num_bins": NUM_BINS,
        "bin_width": BIN_WIDTH,

        "gaussian_sigma_bins": (
            GAUSSIAN_SIGMA_BINS
        ),
        "gaussian_radius_bins": (
            GAUSSIAN_RADIUS_BINS
        ),

        "primary_tolerance_px": (
            PRIMARY_TOLERANCE_PX
        ),

        "seed": SEED,
    }

    torch.save(
        payload,
        path_,
    )


def load_checkpoint(
    path_,
    model,
    optimizer,
    scheduler,
    scaler,
    logger,
):
    checkpoint = torch.load(
        path_,
        map_location=DEVICE,
    )

    if checkpoint.get(
        "model_name"
    ) != MODEL_NAME:
        raise ValueError(
            "Checkpoint MODEL_NAME mismatch."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if scaler_state:
        scaler.load_state_dict(
            scaler_state
        )

    start_epoch = (
        int(
            checkpoint["epoch"]
        )
        + 1
    )

    best_val_f1 = float(
        checkpoint.get(
            "best_val_f1_at_2px",
            -np.inf,
        )
    )

    best_epoch = int(
        checkpoint.get(
            "best_epoch",
            0,
        )
    )

    no_improve_count = int(
        checkpoint.get(
            "no_improve_count",
            0,
        )
    )

    history = checkpoint.get(
        "history",
        [],
    )

    logger.info(
        "Resumed from %s | next_epoch=%d | "
        "best val F1@2px=%.6f | best_epoch=%d",
        path_,
        start_epoch,
        best_val_f1,
        best_epoch,
    )

    return (
        start_epoch,
        best_val_f1,
        best_epoch,
        no_improve_count,
        history,
    )


# ======================================================================
# 13. Run metadata / figures
# ======================================================================

def save_run_metadata(
    model,
    total_params,
    trainable_params,
    normalization_mean,
    normalization_std,
):
    config = {
        "model_name": MODEL_NAME,
        "task": (
            "supervised_localization_82bin_gaussian_heatmap"
        ),

        "seed": SEED,

        "device_policy": (
            "CUDA if available else CPU"
        ),
        "actual_device": str(
            DEVICE
        ),

        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,

        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "geometry": {
            "image_width": IMAGE_WIDTH,
            "num_bins": NUM_BINS,
            "bin_width": BIN_WIDTH,
        },

        "gaussian_heatmap": {
            "sigma_bins": (
                GAUSSIAN_SIGMA_BINS
            ),
            "radius_bins": (
                GAUSSIAN_RADIUS_BINS
            ),
        },

        "loss": {
            "heatmap": (
                "ModifiedFocalLoss"
            ),
            "focal_alpha": (
                FOCAL_ALPHA
            ),
            "focal_beta": (
                FOCAL_BETA
            ),
            "offset": (
                "SmoothL1Loss"
            ),
            "lambda_offset": (
                LAMBDA_OFFSET
            ),
        },

        "localization_head": {
            "fusion_dim": (
                FUSION_DIM
            ),
            "hidden_dim": (
                HEAD_HIDDEN_DIM
            ),
            "dropout": (
                HEAD_DROPOUT
            ),
            "nms_radius_bins": (
                NMS_RADIUS_BINS
            ),
        },

        "evaluation": {
            "primary_metric": (
                "F1@2px"
            ),
            "primary_tolerance_px": (
                PRIMARY_TOLERANCE_PX
            ),
            "checkpoint_threshold": (
                VAL_CHECKPOINT_THRESHOLD
            ),
            "metrics": [
                "Precision@2px",
                "Recall@2px",
                "F1@2px",
                "Acc@1px",
                "Acc@2px",
                "Acc@3px",
                "Top-1 MAE",
                "Matched MAE",
                "False positives/image",
            ],
        },

        "scheduler": {
            "name": (
                "ReduceLROnPlateau"
            ),
            "monitor": "val_loss",
            "factor": LR_FACTOR,
            "patience": LR_PATIENCE,
            "min_lr": MIN_LR,
        },

        "early_stopping": {
            "monitor": (
                "val_f1_at_2px"
            ),
            "mode": "max",
            "patience": (
                EARLY_STOP_PATIENCE
            ),
            "min_delta": (
                MIN_DELTA
            ),
        },

        "normalization": {
            "method": (
                "dataset_zscore"
            ),
            "mean": (
                normalization_mean
            ),
            "std": (
                normalization_std
            ),
            "computed_from": str(
                NORMALIZATION_SOURCE_CSV
            ),
        },

        "data": {
            "train_csv": str(
                TRAIN_CSV
            ),
            "val_csv": str(
                VAL_CSV
            ),
            "test_csv": str(
                TEST_CSV
            ),
        },

        "parameters": {
            "total": int(
                total_params
            ),
            "trainable": int(
                trainable_params
            ),
        },

        "pretrained_weights": False,
        "resize": None,
    }

    with open(
        OUTPUT_ROOT / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUTPUT_ROOT / "model.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(model))
        f.write("\n\n")
        f.write(
            f"Total parameters: {total_params}\n"
        )
        f.write(
            f"Trainable parameters: {trainable_params}\n"
        )


def save_training_figures(
    history_df,
):
    fig = plt.figure(
        figsize=(6.2, 4.4)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train",
        linewidth=1.5,
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation",
        linewidth=1.5,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Localization loss")
    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.45,
    )

    plt.tight_layout()

    fig.savefig(
        FIGURE_DIR / "loss_curves.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig = plt.figure(
        figsize=(6.2, 4.4)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_f1_at_2px"
        ],
        label="Validation",
        linewidth=1.5,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Detection F1 @ 2 px")
    plt.ylim(0.0, 1.02)

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.45,
    )

    plt.tight_layout()

    fig.savefig(
        FIGURE_DIR / "val_f1_curve.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)



# ======================================================================
# 14. Save final test results
# ======================================================================

def save_test_results(
    test_result,
    selected_threshold,
):
    metrics = dict(
        test_result[
            "metrics"
        ]
    )

    metrics.update({
        "loss": float(
            test_result[
                "loss"
            ]
        ),
        "loss_heatmap": float(
            test_result[
                "loss_heatmap"
            ]
        ),
        "loss_offset": float(
            test_result[
                "loss_offset"
            ]
        ),
        "heatmap_threshold": float(
            selected_threshold
        ),
        "primary_tolerance_px": float(
            PRIMARY_TOLERANCE_PX
        ),
        "num_bins": int(
            NUM_BINS
        ),
        "bin_width": float(
            BIN_WIDTH
        ),
    })

    with open(
        METRICS_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    rows = []

    for record in test_result[
        "records"
    ]:
        gt_position = record[
            "gt_position"
        ]

        pred_positions = record[
            "pred_positions"
        ]

        pred_scores = record[
            "pred_scores"
        ]

        top1_position = record[
            "top1_position"
        ]

        if gt_position is not None:
            top1_error = abs(
                float(
                    top1_position
                )
                - float(
                    gt_position
                )
            )
        else:
            top1_error = np.nan

        if (
            gt_position is not None
            and len(pred_positions) > 0
        ):
            errors = np.abs(
                np.asarray(
                    pred_positions,
                    dtype=np.float64,
                )
                - float(
                    gt_position
                )
            )

            nearest_error = float(
                errors.min()
            )
        else:
            nearest_error = np.nan

        rows.append({
            "file_name": record[
                "file_name"
            ],
            "label": record[
                "label"
            ],
            "gt_position": (
                gt_position
                if gt_position is not None
                else np.nan
            ),

            "top1_position": float(
                top1_position
            ),
            "top1_score": float(
                record[
                    "top1_score"
                ]
            ),
            "top1_error_px": (
                top1_error
            ),

            "num_predictions": len(
                pred_positions
            ),

            "pred_positions": json.dumps(
                [
                    round(
                        float(x),
                        6,
                    )
                    for x in pred_positions
                ]
            ),

            "pred_scores": json.dumps(
                [
                    round(
                        float(x),
                        6,
                    )
                    for x in pred_scores
                ]
            ),

            "nearest_error_px": (
                nearest_error
            ),

            "hit_at_1px": (
                bool(
                    np.isfinite(
                        nearest_error
                    )
                    and nearest_error <= 1.0
                )
                if gt_position is not None
                else False
            ),

            "hit_at_2px": (
                bool(
                    np.isfinite(
                        nearest_error
                    )
                    and nearest_error <= 2.0
                )
                if gt_position is not None
                else False
            ),

            "hit_at_3px": (
                bool(
                    np.isfinite(
                        nearest_error
                    )
                    and nearest_error <= 3.0
                )
                if gt_position is not None
                else False
            ),
        })

    pd.DataFrame(
        rows
    ).to_csv(
        METRICS_DIR
        / "test_predictions.csv",
        index=False,
    )


# ======================================================================
# 15. Main
# ======================================================================

def main():
    prepare_output_dirs()

    if not RESUME and BATCH_LOSS_CSV.exists():
        BATCH_LOSS_CSV.unlink()

    logger = build_logger()
    set_seed(SEED)

    logger.info("=" * 78)
    logger.info(
        "Dark Soliton Supervised Localization Baseline"
    )
    logger.info(
        "82-bin Gaussian Heatmap + Offset Regression"
    )
    logger.info("=" * 78)

    logger.info(
        "Backbone      : %s",
        MODEL_NAME,
    )
    logger.info(
        "Device        : %s",
        DEVICE,
    )
    logger.info(
        "PyTorch       : %s",
        torch.__version__,
    )
    logger.info(
        "CUDA avail.   : %s",
        torch.cuda.is_available(),
    )

    if DEVICE.type == "cuda":
        logger.info(
            "CUDA device   : %s",
            torch.cuda.get_device_name(
                0
            ),
        )

    if MODEL_NAME in {
        "repvit_m0_9",
        "fasternet_t0",
    }:
        timm = _import_timm()
        logger.info(
            "timm          : %s",
            timm.__version__,
        )

    logger.info(
        "Image width   : %d px",
        IMAGE_WIDTH,
    )
    logger.info(
        "Localization  : %d bins x %.1f px/bin",
        NUM_BINS,
        BIN_WIDTH,
    )
    logger.info(
        "Gaussian      : sigma=%.2f bins, radius=%d bins",
        GAUSSIAN_SIGMA_BINS,
        GAUSSIAN_RADIUS_BINS,
    )
    logger.info(
        "Primary rule  : +/- %.1f px",
        PRIMARY_TOLERANCE_PX,
    )
    logger.info(
        "Batch size    : %d",
        BATCH_SIZE,
    )
    logger.info(
        "Num workers   : %d",
        NUM_WORKERS,
    )

    normalization_mean, normalization_std = (
        compute_project_normalization(
            logger
        )
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_datasets_and_loaders(
        normalization_mean,
        normalization_std,
    )

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

    model = DarkSolitonDetector(
        MODEL_NAME
    ).to(
        DEVICE
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    logger.info(
        "Parameters    : total=%s, trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    heatmap_criterion = ModifiedFocalLoss(
        alpha=FOCAL_ALPHA,
        beta=FOCAL_BETA,
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

    amp_enabled = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    logger.info(
        "AMP enabled   : %s",
        amp_enabled,
    )

    save_run_metadata(
        model=model,
        total_params=total_params,
        trainable_params=trainable_params,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    start_epoch = 1
    best_val_f1 = -np.inf
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
            best_val_f1,
            best_epoch,
            no_improve_count,
            history,
        ) = load_checkpoint(
            path_=LAST_CKPT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            logger=logger,
        )

    training_start = time.time()

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.time()

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            heatmap_criterion=heatmap_criterion,
            offset_criterion=offset_criterion,
            epoch=epoch,
        )

        pd.DataFrame(
            train_result["batch_records"]
        ).to_csv(
            BATCH_LOSS_CSV,
            mode="a",
            header=not BATCH_LOSS_CSV.exists(),
            index=False,
        )

        val_result = evaluate_detector(
            model=model,
            loader=val_loader,
            heatmap_criterion=heatmap_criterion,
            offset_criterion=offset_criterion,
            threshold=VAL_CHECKPOINT_THRESHOLD,
        )

        scheduler.step(
            val_result[
                "loss"
            ]
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        val_metrics = val_result[
            "metrics"
        ]

        current_val_f1 = float(
            val_metrics[
                "f1_at_2px"
            ]
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": (
                train_result[
                    "loss"
                ]
            ),
            "train_loss_heatmap": (
                train_result[
                    "loss_heatmap"
                ]
            ),
            "train_loss_offset": (
                train_result[
                    "loss_offset"
                ]
            ),

            "val_loss": (
                val_result[
                    "loss"
                ]
            ),
            "val_loss_heatmap": (
                val_result[
                    "loss_heatmap"
                ]
            ),
            "val_loss_offset": (
                val_result[
                    "loss_offset"
                ]
            ),

            "val_precision_at_2px": (
                val_metrics[
                    "precision_at_2px"
                ]
            ),
            "val_recall_at_2px": (
                val_metrics[
                    "recall_at_2px"
                ]
            ),
            "val_f1_at_2px": (
                val_metrics[
                    "f1_at_2px"
                ]
            ),

            "val_acc_at_1px": (
                val_metrics[
                    "acc_at_1px"
                ]
            ),
            "val_acc_at_2px": (
                val_metrics[
                    "acc_at_2px"
                ]
            ),
            "val_acc_at_3px": (
                val_metrics[
                    "acc_at_3px"
                ]
            ),

            "val_top1_mae_px": (
                val_metrics[
                    "top1_mae_px"
                ]
            ),
            "val_matched_mae_px": (
                val_metrics[
                    "matched_mae_px"
                ]
            ),

            "val_false_positives_per_image": (
                val_metrics[
                    "false_positives_per_image"
                ]
            ),

            "epoch_seconds": (
                epoch_seconds
            ),
        }

        history.append(
            row
        )

        improved = (
            current_val_f1
            > (
                best_val_f1
                + MIN_DELTA
            )
        )

        if improved:
            best_val_f1 = (
                current_val_f1
            )
            best_epoch = epoch
            no_improve_count = 0

            save_checkpoint(
                path_=BEST_CKPT,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                no_improve_count=no_improve_count,
                history=history,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
            )
        else:
            no_improve_count += 1

        save_checkpoint(
            path_=LAST_CKPT,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            no_improve_count=no_improve_count,
            history=history,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            METRICS_DIR
            / "training_history.csv",
            index=False,
        )

        marker = (
            " *BEST*"
            if improved
            else ""
        )

        logger.info(
            "Epoch %03d/%03d | "
            "lr=%.2e | "
            "train=%.5f (hm=%.5f,off=%.5f) | "
            "val=%.5f | "
            "P@2=%.4f R@2=%.4f F1@2=%.4f | "
            "Acc@1/2/3=%.4f/%.4f/%.4f | "
            "Top1MAE=%.3f MatchedMAE=%.3f | "
            "%.1fs%s",
            epoch,
            MAX_EPOCHS,
            current_lr,

            train_result[
                "loss"
            ],
            train_result[
                "loss_heatmap"
            ],
            train_result[
                "loss_offset"
            ],

            val_result[
                "loss"
            ],

            val_metrics[
                "precision_at_2px"
            ],
            val_metrics[
                "recall_at_2px"
            ],
            val_metrics[
                "f1_at_2px"
            ],

            val_metrics[
                "acc_at_1px"
            ],
            val_metrics[
                "acc_at_2px"
            ],
            val_metrics[
                "acc_at_3px"
            ],

            val_metrics[
                "top1_mae_px"
            ],
            val_metrics[
                "matched_mae_px"
            ],

            epoch_seconds,
            marker,
        )

        if (
            no_improve_count
            >= EARLY_STOP_PATIENCE
        ):
            logger.info(
                "Early stopping at epoch %d: "
                "val F1@2px did not improve by > %.6f "
                "for %d epochs.",
                epoch,
                MIN_DELTA,
                EARLY_STOP_PATIENCE,
            )
            break

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        METRICS_DIR
        / "training_history.csv",
        index=False,
    )

    save_training_figures(
        history_df
    )

    training_seconds = (
        time.time()
        - training_start
    )

    logger.info(
        "Training finished | %.2f min",
        training_seconds
        / 60.0,
    )

    logger.info(
        "Best epoch=%d | "
        "best val F1@2px (threshold=%.2f)=%.6f",
        best_epoch,
        VAL_CHECKPOINT_THRESHOLD,
        best_val_f1,
    )

    if not BEST_CKPT.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CKPT}"
        )

    best_checkpoint = torch.load(
        BEST_CKPT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    (
        selected_threshold,
        selected_val_metrics,
        _,
    ) = search_best_threshold(
        model=model,
        val_loader=val_loader,
        heatmap_criterion=heatmap_criterion,
        offset_criterion=offset_criterion,
    )

    logger.info(
        "Selected validation threshold=%.2f | "
        "Val P@2=%.4f R@2=%.4f F1@2=%.4f | "
        "Top1MAE=%.3f px",
        selected_threshold,
        selected_val_metrics[
            "precision_at_2px"
        ],
        selected_val_metrics[
            "recall_at_2px"
        ],
        selected_val_metrics[
            "f1_at_2px"
        ],
        selected_val_metrics[
            "top1_mae_px"
        ],
    )

    test_result = evaluate_detector(
        model=model,
        loader=test_loader,
        heatmap_criterion=heatmap_criterion,
        offset_criterion=offset_criterion,
        threshold=selected_threshold,
    )

    save_test_results(
        test_result=test_result,
        selected_threshold=selected_threshold,
    )

    metrics = test_result[
        "metrics"
    ]

    logger.info("=" * 78)
    logger.info(
        "FINAL TEST RESULTS — BEST CHECKPOINT"
    )
    logger.info(
        "82-bin Gaussian Heatmap + Offset Regression"
    )
    logger.info("=" * 78)

    logger.info(
        "Selected threshold       : %.2f",
        selected_threshold,
    )

    logger.info(
        "Test loss                : %.6f",
        test_result[
            "loss"
        ],
    )

    logger.info(
        "Heatmap loss             : %.6f",
        test_result[
            "loss_heatmap"
        ],
    )

    logger.info(
        "Offset loss              : %.6f",
        test_result[
            "loss_offset"
        ],
    )

    logger.info(
        "Precision @ 2 px         : %.6f",
        metrics[
            "precision_at_2px"
        ],
    )

    logger.info(
        "Recall @ 2 px            : %.6f",
        metrics[
            "recall_at_2px"
        ],
    )

    logger.info(
        "F1 @ 2 px                : %.6f",
        metrics[
            "f1_at_2px"
        ],
    )

    logger.info(
        "Acc @ 1 px               : %.6f",
        metrics[
            "acc_at_1px"
        ],
    )

    logger.info(
        "Acc @ 2 px               : %.6f",
        metrics[
            "acc_at_2px"
        ],
    )

    logger.info(
        "Acc @ 3 px               : %.6f",
        metrics[
            "acc_at_3px"
        ],
    )

    logger.info(
        "Top-1 MAE                : %.6f px",
        metrics[
            "top1_mae_px"
        ],
    )

    logger.info(
        "Matched MAE              : %.6f px",
        metrics[
            "matched_mae_px"
        ],
    )

    logger.info(
        "False positives / image  : %.6f",
        metrics[
            "false_positives_per_image"
        ],
    )

    logger.info(
        "Mean predictions / image : %.6f",
        metrics[
            "mean_predictions_per_image"
        ],
    )

    logger.info(
        "Output directory         : %s",
        OUTPUT_ROOT,
    )

    logger.info("=" * 78)


if __name__ == "__main__":
    main()

