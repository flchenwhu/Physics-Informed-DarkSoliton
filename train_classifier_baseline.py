# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier_baseline.py
============================

Dark Soliton 三分类监督 Baseline 训练脚本。
"""

from pathlib import Path
import json
import logging
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18

# timm 仅在 RepViT / FasterNet 被选择时需要。
# 这里不做顶层强制 import，保证未安装 timm 时仍可运行 simple_cnn / resnet18。

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


# ======================================================================
# 1. 用户配置
# ======================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# 模型切换：只改这里
# MODEL_NAME = "simple_cnn"
# MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
MODEL_NAME = "fasternet_t0"

NUM_CLASSES = 3
CLASS_NAMES = [
    "no_excitation",
    "single_excitation",
    "other_excitations",
]

SEED = 2027    # 跑3轮随机种子   取mean+std  (42  90  2027)

BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 70
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

USE_CLASS_WEIGHTS = True
GRAD_CLIP_NORM = 5.0

# CUDA 时启用 AMP；CPU 自动关闭
USE_AMP = True

# Scheduler
LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6

# Early stopping
EARLY_STOP_PATIENCE = 15
MIN_DELTA = 1e-4

# Resume
RESUME = False

# 数据检查
CHECK_FILE_EXISTS = True

# 输出目录
OUTPUT_ROOT = ROOT / "output" / MODEL_NAME / "classifier_baseline"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoint"
LOG_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

BEST_CKPT = CHECKPOINT_DIR / "best.pt.tar"
LAST_CKPT = CHECKPOINT_DIR / "last.pt.tar"
LOG_FILE = LOG_DIR / "trainer.log"

# 每个 training batch 的原始 loss
BATCH_LOSS_CSV = METRICS_DIR / "batch_loss_history.csv"


# ======================================================================
# 2. Device：只判断 CUDA / CPU
# ======================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 3. Reproducibility
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


# ======================================================================
# 4. Logging / output
# ======================================================================

def prepare_output_dirs():
    for path in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_logger():
    logger = logging.getLogger("dark_soliton_classifier")
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

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ======================================================================
# 5. Models
# ======================================================================

class SimpleCNN(nn.Module):
    """简洁的三分类 CNN Baseline，不是原论文 CNN 的逐层复现。"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _import_timm():
    """
    仅在使用 RepViT / FasterNet 时加载 timm。

    这样即使环境没有安装 timm，也不会影响 simple_cnn / resnet18。
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "MODEL_NAME 选择了 RepViT/FasterNet，但当前环境未安装 timm。\n"
            "建议在已激活的 darksoliton 环境中安装：\n"
            "    mamba install timm\n"
            "安装后可验证：\n"
            "    python -c \"import timm; print(timm.__version__)\""
        ) from exc

    return timm


def build_model(model_name, num_classes):
    """
    构建四个监督分类 Baseline。

    公平性原则：
    - 全部 pretrained=False；
    - 全部直接接受 1-channel 输入；
    - 全部输出 NUM_CLASSES=3；
    - 不 resize，输入仍为 [B, 1, 132, 164]。
    """
    model_name = model_name.lower().strip()

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name in {"repvit_m0_9", "fasternet_t0"}:
        timm = _import_timm()

        # timm 原生支持 in_chans / num_classes 参数。
        # 这里不加载 ImageNet 权重，保持与 simple_cnn / resnet18 的
        # "from scratch" 监督 Baseline 条件一致。
        model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1,
            num_classes=num_classes,
        )

        return model

    raise ValueError(
        f"Unsupported MODEL_NAME={model_name!r}. "
        "Use one of: "
        "'simple_cnn', 'resnet18', 'repvit_m0_9', 'fasternet_t0'."
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def forward_with_penultimate_features(model, images):
    """
    一次 backbone 前向中同时获得：
        logits
        penultimate features

    用于最终 best-checkpoint test 的特征导出。

    Returns
    -------
    logits:
        [B, NUM_CLASSES]

    features:
        [B, D]

    其中 D：
    - SimpleCNN = 128
    - ResNet18 = 512
    - RepViT / FasterNet = timm 模型自身的 pre-logits feature dimension
    """

    # --------------------------------------------------------------
    # SimpleCNN
    # --------------------------------------------------------------
    if isinstance(model, SimpleCNN):
        x = model.features(images)
        x = model.pool(x)
        features = torch.flatten(x, 1)
        logits = model.classifier(features)
        return logits, features

    # --------------------------------------------------------------
    # torchvision ResNet18
    # 注意：必须在 timm 通用分支之前判断。
    # --------------------------------------------------------------
    if (
        hasattr(model, "layer1")
        and hasattr(model, "layer4")
        and hasattr(model, "avgpool")
        and hasattr(model, "fc")
        and not hasattr(model, "forward_head")
    ):
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = model.fc(features)

        return logits, features

    # --------------------------------------------------------------
    # timm models: RepViT / FasterNet
    #
    # timm 统一接口：
    #   forward_features()       -> backbone feature map
    #   forward_head(..., True)  -> pre-logits / penultimate feature
    #   forward_head(..., False) -> logits
    #
    # 最终 test 时 model.eval()，因此 head dropout 不会引入随机性。
    # --------------------------------------------------------------
    if (
        hasattr(model, "forward_features")
        and hasattr(model, "forward_head")
    ):
        feature_map = model.forward_features(images)

        features = model.forward_head(
            feature_map,
            pre_logits=True,
        )

        logits = model.forward_head(
            feature_map,
            pre_logits=False,
        )

        # 某些模型的 pre-logits 可能仍保留额外维度，
        # t-SNE 需要 [B, D]，因此统一展平。
        if features.ndim > 2:
            features = torch.flatten(
                features,
                start_dim=1,
            )

        return logits, features

    raise TypeError(
        "Unsupported model type for penultimate feature extraction: "
        f"{type(model).__name__}"
    )


# ======================================================================
# 6. Data
# ======================================================================

def compute_train_normalization(logger):
    logger.info("Computing normalization statistics from TRAIN SET only...")

    raw_train_dataset = DarkSolitonDataset(
        csv_path=TRAIN_CSV,
        task="classification",
        normalize="none",
        check_file_exists=CHECK_FILE_EXISTS,
    )

    mean, std = compute_dataset_mean_std(
        raw_train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    if std <= 0:
        raise RuntimeError(f"Invalid train std={std}")

    normalization = {
        "method": "dataset_zscore",
        "mean": mean,
        "std": std,
        "source": str(TRAIN_CSV),
        "note": "Computed from train.csv only; validation/test were not used.",
    }

    with open(METRICS_DIR / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(normalization, f, indent=2, ensure_ascii=False)

    logger.info("Train normalization: mean=%.8f, std=%.8f", mean, std)
    return mean, std


def build_datasets_and_loaders(mean, std):
    common_kwargs = dict(
        task="classification",
        normalize="dataset_zscore",
        dataset_mean=mean,
        dataset_std=std,
        check_file_exists=CHECK_FILE_EXISTS,
    )

    train_dataset = DarkSolitonDataset(csv_path=TRAIN_CSV, **common_kwargs)
    val_dataset = DarkSolitonDataset(csv_path=VAL_CSV, **common_kwargs)
    test_dataset = DarkSolitonDataset(csv_path=TEST_CSV, **common_kwargs)

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


def compute_class_weights(train_dataset):
    counts_dict = train_dataset.class_counts()
    counts = np.array(
        [counts_dict.get(i, 0) for i in range(NUM_CLASSES)],
        dtype=np.float64,
    )

    if np.any(counts <= 0):
        raise RuntimeError(f"Invalid class counts: {counts.tolist()}")

    total = counts.sum()
    weights = total / (NUM_CLASSES * counts)

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ======================================================================
# 7. Metrics
# ======================================================================

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


# ======================================================================
# 8. Train / evaluate
# ======================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    amp_enabled,
    epoch,
):
    model.train()

    running_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []
    batch_records = []

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            DEVICE, non_blocking=(DEVICE.type == "cuda")
        )
        labels = batch["label"].to(
            DEVICE, non_blocking=(DEVICE.type == "cuda")
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        batch_records.append({
            "epoch": int(epoch),
            "batch": int(batch_index),
            "global_step": int(
                (epoch - 1) * len(loader) + batch_index
            ),
            "batch_size": int(batch_size),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": float(loss.detach().item()),
        })

        predictions = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(predictions.detach().cpu().tolist())

    epoch_loss = running_loss / total_samples
    metrics = compute_metrics(all_true, all_pred)
    return epoch_loss, metrics, batch_records


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    collect_predictions=False,
    collect_features=False,
):
    """
    验证 / 测试。

    collect_predictions=True:
        收集 file_name、概率等预测信息。

    collect_features=True:
        提取 penultimate feature。
        建议仅在最终 test 时开启。
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []
    all_probs = []
    all_file_names = []
    all_features = []

    for batch in loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )
        labels = batch["label"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        if collect_features:
            logits, features = forward_with_penultimate_features(
                model,
                images,
            )
        else:
            features = None
            logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )
        predictions = probabilities.argmax(
            dim=1
        )

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_true.extend(
            labels.cpu().tolist()
        )
        all_pred.extend(
            predictions.cpu().tolist()
        )

        if collect_predictions:
            all_probs.append(
                probabilities.cpu().numpy()
            )
            all_file_names.extend(
                batch["file_name"]
            )

        if collect_features:
            all_features.append(
                features.cpu().numpy()
            )

    epoch_loss = running_loss / total_samples
    metrics = compute_metrics(
        all_true,
        all_pred,
    )

    result = {
        "loss": epoch_loss,
        "metrics": metrics,
        "y_true": np.asarray(
            all_true,
            dtype=np.int64,
        ),
        "y_pred": np.asarray(
            all_pred,
            dtype=np.int64,
        ),
    }

    if collect_predictions:
        result["probabilities"] = np.concatenate(
            all_probs,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )
        result["file_names"] = list(
            all_file_names
        )

    if collect_features:
        result["features"] = np.concatenate(
            all_features,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

    return result


# ======================================================================
# 9. Checkpoint
# ======================================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_macro_f1,
    best_epoch,
    no_improve_count,
    history,
    normalization_mean,
    normalization_std,
):
    checkpoint = {
        "epoch": int(epoch),
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_epoch": int(best_epoch),
        "no_improve_count": int(no_improve_count),
        "normalization": {
            "method": "dataset_zscore",
            "mean": float(normalization_mean),
            "std": float(normalization_std),
        },
        "history": history,
        "seed": SEED,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, logger):
    checkpoint = torch.load(path, map_location=DEVICE)

    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != MODEL_NAME:
        raise ValueError(
            "Checkpoint model_name mismatch: "
            f"{checkpoint_model_name!r} != {MODEL_NAME!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -np.inf))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    no_improve_count = int(checkpoint.get("no_improve_count", 0))
    history = checkpoint.get("history", [])

    logger.info(
        "Resumed from %s | next_epoch=%d | best_val_macro_f1=%.6f | best_epoch=%d",
        path,
        start_epoch,
        best_val_macro_f1,
        best_epoch,
    )

    return (
        start_epoch,
        best_val_macro_f1,
        best_epoch,
        no_improve_count,
        history,
    )


# ======================================================================
# 10. Save config / model
# ======================================================================

def save_run_metadata(
    model,
    total_params,
    trainable_params,
    normalization_mean,
    normalization_std,
    class_weights,
):
    config = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "device_policy": "CUDA if available else CPU",
        "actual_device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "class_weights": None if class_weights is None else [
            float(x) for x in class_weights.detach().cpu().tolist()
        ],
        "grad_clip_norm": GRAD_CLIP_NORM,
        "use_amp": USE_AMP,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": LR_FACTOR,
            "patience": LR_PATIENCE,
            "min_lr": MIN_LR,
            "monitor": "val_loss",
        },
        "early_stopping": {
            "monitor": "val_macro_f1",
            "mode": "max",
            "patience": EARLY_STOP_PATIENCE,
            "min_delta": MIN_DELTA,
        },
        "normalization": {
            "method": "dataset_zscore",
            "mean": normalization_mean,
            "std": normalization_std,
            "computed_from": str(TRAIN_CSV),
        },
        "data": {
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "test_csv": str(TEST_CSV),
        },
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
        },
        "resize": None,
        "augmentation": None,
        "pretrained_weights": False,
        "model_library": (
            "timm"
            if MODEL_NAME in {"repvit_m0_9", "fasternet_t0"}
            else "torchvision/project"
        ),
    }

    with open(OUTPUT_ROOT / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_ROOT / "model.txt", "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")


# ======================================================================
# 11. Figures
# ======================================================================

def save_training_figures(history_df):
    fig = plt.figure(figsize=(6.2, 4.4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train", linewidth=1.5)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 4.4))
    plt.plot(history_df["epoch"], history_df["train_macro_f1"], label="Train", linewidth=1.5)
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="Validation", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "macro_f1_curve.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(cm):
    fig = plt.figure(figsize=(5.4, 4.8))
    ax = plt.gca()
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=["0", "1", "2"],
        yticklabels=["0", "1", "2"],
        ylabel="True label",
        xlabel="Predicted label",
    )

    threshold = cm.max() / 2.0 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 12. Test result saving
# ======================================================================

def save_test_results(test_result):
    y_true = test_result["y_true"]
    y_pred = test_result["y_pred"]
    probabilities = test_result["probabilities"]
    file_names = test_result["file_names"]

    metrics = dict(test_result["metrics"])
    metrics["cross_entropy_loss"] = float(test_result["loss"])

    with open(METRICS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR / "test_classification_report.csv"
    )

    predictions_df = pd.DataFrame({
        "file_name": file_names,
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_class0": probabilities[:, 0],
        "prob_class1": probabilities[:, 1],
        "prob_class2": probabilities[:, 2],
    })
    predictions_df.to_csv(METRICS_DIR / "test_predictions.csv", index=False)


    # Penultimate features for t-SNE / UMAP
    features = test_result["features"]

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D features, got shape={features.shape}"
        )

    if features.shape[0] != len(file_names):
        raise ValueError(
            "Feature rows and file_names have inconsistent lengths: "
            f"{features.shape[0]} vs {len(file_names)}"
        )

    feature_columns = [
        f"feature_{i:03d}"
        for i in range(features.shape[1])
    ]

    features_df = pd.DataFrame(
        features,
        columns=feature_columns,
    )

    # 只保留 file_name 作为唯一关联键。
    # 标签/预测/概率已保存在 test_predictions.csv。
    features_df.insert(
        0,
        "file_name",
        file_names,
    )

    features_df.to_csv(
        METRICS_DIR / "test_features.csv",
        index=False,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )
    cm_df.to_csv(METRICS_DIR / "confusion_matrix.csv")
    save_confusion_matrix_figure(cm)


# ======================================================================
# 13. Main
# ======================================================================

def main():
    prepare_output_dirs()

    if not RESUME and BATCH_LOSS_CSV.exists():
        BATCH_LOSS_CSV.unlink()

    logger = build_logger()
    set_seed(SEED)

    logger.info("=" * 78)
    logger.info("Dark Soliton Classification Baseline")
    logger.info("=" * 78)
    logger.info("Model       : %s", MODEL_NAME)

    if MODEL_NAME in {"repvit_m0_9", "fasternet_t0"}:
        timm = _import_timm()
        logger.info("timm        : %s", timm.__version__)

    logger.info("Device      : %s", DEVICE)
    logger.info("PyTorch     : %s", torch.__version__)
    logger.info("CUDA avail. : %s", torch.cuda.is_available())

    if DEVICE.type == "cuda":
        logger.info("CUDA device : %s", torch.cuda.get_device_name(0))

    logger.info("Batch size  : %d", BATCH_SIZE)
    logger.info("Num workers : %d", NUM_WORKERS)

    normalization_mean, normalization_std = compute_train_normalization(logger)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_datasets_and_loaders(normalization_mean, normalization_std)

    logger.info("Train: N=%d, counts=%s", len(train_dataset), train_dataset.class_counts())
    logger.info("Val  : N=%d, counts=%s", len(val_dataset), val_dataset.class_counts())
    logger.info("Test : N=%d, counts=%s", len(test_dataset), test_dataset.class_counts())

    model = build_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    logger.info(
        "Parameters  : total=%s, trainable=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_dataset)
        logger.info(
            "Class weights: %s",
            [round(float(x), 6) for x in class_weights.detach().cpu().tolist()],
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    logger.info("AMP enabled : %s", amp_enabled)

    save_run_metadata(
        model=model,
        total_params=total_params,
        trainable_params=trainable_params,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        class_weights=class_weights,
    )

    start_epoch = 1
    best_val_macro_f1 = -np.inf
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
            best_val_macro_f1,
            best_epoch,
            no_improve_count,
            history,
        ) = load_checkpoint(
            path=LAST_CKPT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            logger=logger,
        )

    training_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()

        (
            train_loss,
            train_metrics,
            batch_records,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            epoch=epoch,
        )

        pd.DataFrame(batch_records).to_csv(
            BATCH_LOSS_CSV,
            mode="a",
            header=not BATCH_LOSS_CSV.exists(),
            index=False,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            collect_predictions=False,
        )

        val_loss = float(val_result["loss"])
        val_metrics = val_result["metrics"]
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        current_val_macro_f1 = val_metrics["macro_f1"]
        improved = current_val_macro_f1 > (best_val_macro_f1 + MIN_DELTA)

        if improved:
            best_val_macro_f1 = current_val_macro_f1
            best_epoch = epoch
            no_improve_count = 0

            save_checkpoint(
                path=BEST_CKPT,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_macro_f1=best_val_macro_f1,
                best_epoch=best_epoch,
                no_improve_count=no_improve_count,
                history=history,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
            )
        else:
            no_improve_count += 1

        save_checkpoint(
            path=LAST_CKPT,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_macro_f1=best_val_macro_f1,
            best_epoch=best_epoch,
            no_improve_count=no_improve_count,
            history=history,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(METRICS_DIR / "training_history.csv", index=False)

        marker = " *BEST*" if improved else ""
        logger.info(
            "Epoch %03d/%03d | lr=%.2e | "
            "train loss=%.5f acc=%.4f macroF1=%.4f | "
            "val loss=%.5f acc=%.4f macroF1=%.4f | %.1fs%s",
            epoch,
            MAX_EPOCHS,
            current_lr,
            train_loss,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            epoch_seconds,
            marker,
        )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping at epoch %d: val Macro-F1 did not improve "
                "by > %.6f for %d epochs.",
                epoch,
                MIN_DELTA,
                EARLY_STOP_PATIENCE,
            )
            break

    training_seconds = time.time() - training_start

    history_df = pd.DataFrame(history)
    history_df.to_csv(METRICS_DIR / "training_history.csv", index=False)
    save_training_figures(history_df)

    logger.info("Training finished | %.2f min", training_seconds / 60.0)
    logger.info(
        "Best epoch=%d | best val Macro-F1=%.6f",
        best_epoch,
        best_val_macro_f1,
    )

    if not BEST_CKPT.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {BEST_CKPT}")

    best_checkpoint = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_result = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        collect_predictions=True,
        collect_features=True,
    )
    save_test_results(test_result)

    logger.info(
        "Test penultimate features: shape=%s -> %s",
        tuple(test_result["features"].shape),
        METRICS_DIR / "test_features.csv",
    )

    test_metrics = test_result["metrics"]

    logger.info("=" * 78)
    logger.info("FINAL TEST RESULTS — BEST CHECKPOINT")
    logger.info("=" * 78)
    logger.info("Test loss              : %.6f", test_result["loss"])
    logger.info("Accuracy               : %.6f", test_metrics["accuracy"])
    logger.info("Balanced Accuracy      : %.6f", test_metrics["balanced_accuracy"])
    logger.info("Macro Precision        : %.6f", test_metrics["macro_precision"])
    logger.info("Macro Recall           : %.6f", test_metrics["macro_recall"])
    logger.info("Macro F1               : %.6f", test_metrics["macro_f1"])
    logger.info("Weighted F1            : %.6f", test_metrics["weighted_f1"])
    logger.info("Output directory       : %s", OUTPUT_ROOT)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
