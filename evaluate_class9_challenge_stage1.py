# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_class9_challenge_stage1.py
===================================

Stage 1 supervised classifier 在 class-9 人工 Gold 数据上的最终泛化测试。
"""

from pathlib import Path
import json
import logging
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import train_classifier_baseline as stage1_module
from dark_soliton_dataloader import DarkSolitonDataset, make_dataloader


# =============================================================================
# 1. Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

GOLD_CSV = DATA_DIR / "class9_inference2_manual_annotations.csv"

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

BATCH_SIZE = 32
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_OUTPUT_ROOT = (
    ROOT / "output" / MODEL_NAME / "classifier_baseline"
)

CHECKPOINT_PATH = (
    MODEL_OUTPUT_ROOT / "checkpoint" / "best.pt.tar"
)

OUTPUT_DIR = (
    MODEL_OUTPUT_ROOT / "gold_challenge_class9"
)

PREDICTIONS_CSV = OUTPUT_DIR / "class9_gold_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "class9_gold_metrics.json"
METRICS_CSV = OUTPUT_DIR / "class9_gold_metrics.csv"
REPORT_CSV = OUTPUT_DIR / "class9_gold_classification_report.csv"
CONFUSION_CSV = OUTPUT_DIR / "class9_gold_confusion_matrix.csv"
PAPER_CORE_JSON = OUTPUT_DIR / "paper_core_metrics.json"
PAPER_CORE_CSV = OUTPUT_DIR / "paper_core_metrics.csv"

# Penultimate Gold features for t-SNE / UMAP.
GOLD_FEATURES_CSV = OUTPUT_DIR / "gold_features.csv"


# =============================================================================
# 2. Logging
# =============================================================================

def build_logger():
    logger = logging.getLogger("stage1_class9_gold")
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


# =============================================================================
# 3. Gold loading and validation
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
        "manual_label",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            f"Gold CSV missing columns: {sorted(missing)}"
        )

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Gold CSV is empty.")

    duplicate = (
        df["file_name"]
        .astype(str)
        .duplicated(keep=False)
    )

    if duplicate.any():
        examples = (
            df.loc[duplicate, "file_name"]
            .astype(str)
            .tolist()[:10]
        )
        raise ValueError(
            "Gold CSV contains duplicated file_name values. "
            f"Examples: {examples}"
        )

    df["manual_label"] = pd.to_numeric(
        df["manual_label"],
        errors="raise",
    ).astype(int)

    invalid = sorted(
        set(df["manual_label"].tolist())
        - {-1, 0, 1, 2}
    )

    if invalid:
        raise ValueError(
            f"Invalid manual_label values: {invalid}"
        )

    df["_included_classification"] = (
        df["manual_label"].isin([0, 1, 2])
    )

    included = df[
        df["_included_classification"]
    ]

    if len(included) == 0:
        raise RuntimeError(
            "No non-ambiguous Gold labels are available."
        )

    present_labels = set(
        included["manual_label"].tolist()
    )

    missing_classes = {0, 1, 2} - present_labels
    if missing_classes:
        raise ValueError(
            "Class-9 Gold classification subset does not contain all "
            f"three classes. Missing: {sorted(missing_classes)}"
        )

    return df


# =============================================================================
# 4. Metrics
# =============================================================================

def compute_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

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
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    return {
        "num_images": int(len(y_true)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
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
# 5. Inference
# =============================================================================

@torch.no_grad()
def run_inference(model, loader, gold_df):
    model.eval()
    records = []
    feature_records = []

    for batch in loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        # Original evaluation forward:
        # logits = model(images)
        #
        # Use the training script's existing helper so predictions and
        # penultimate classification features come from the SAME forward.
        logits, penultimate_features = (
            stage1_module
            .forward_with_penultimate_features(
                model,
                images,
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()

        indices = batch["index"].cpu().tolist()

        for batch_index, row_index in enumerate(indices):
            row = gold_df.iloc[int(row_index)]
            probs = probabilities[batch_index]

            records.append(
                {
                    "file_name": str(row["file_name"]),
                    "manual_label": int(row["manual_label"]),
                    "included_classification": bool(
                        row["_included_classification"]
                    ),
                    "pred_class": int(np.argmax(probs)),
                    "prob_class0": float(probs[0]),
                    "prob_class1": float(probs[1]),
                    "prob_class2": float(probs[2]),
                }
            )

            feature_vector = (
                penultimate_features[
                    batch_index
                ]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            feature_row = {
                "file_name": str(row["file_name"]),
                "manual_label": int(row["manual_label"]),
                "included_classification": bool(
                    row["_included_classification"]
                ),
                "pred_class": int(np.argmax(probs)),
                "feature_type": (
                    "penultimate_classification_feature"
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
# 6. Saving
# =============================================================================

def save_full_metrics(metrics):
    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    rows = []

    scalar_keys = [
        "num_images",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    for key in scalar_keys:
        rows.append(
            {
                "scope": "overall",
                "metric": key,
                "value": metrics[key],
                "role": (
                    "PAPER_CORE"
                    if key in {
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                    }
                    else "AUXILIARY"
                ),
            }
        )

    for label in [0, 1, 2]:
        class_metrics = metrics["per_class"][str(label)]

        for key in [
            "precision",
            "recall",
            "f1",
            "support",
        ]:
            rows.append(
                {
                    "scope": f"class_{label}",
                    "metric": key,
                    "value": class_metrics[key],
                    "role": "AUXILIARY",
                }
            )

    pd.DataFrame(rows).to_csv(
        METRICS_CSV,
        index=False,
    )


def save_paper_core(metrics):
    payload = {
        "task": "class9_heldout_classification",
        "model_stage": "stage1_classifier",
        "model_name": MODEL_NAME,
        "num_images": metrics["num_images"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "note": (
            "manual_label=-1 samples are excluded. "
            "These are the recommended primary classification "
            "metrics for the paper."
        ),
    }

    with open(
        PAPER_CORE_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        [
            {
                "task": "class9_classification",
                "subset": "class9_inference2_gold",
                "metric": "accuracy",
                "value": metrics["accuracy"],
                "role": "PAPER_CORE",
            },
            {
                "task": "class9_classification",
                "subset": "class9_inference2_gold",
                "metric": "balanced_accuracy",
                "value": metrics["balanced_accuracy"],
                "role": "PAPER_CORE",
            },
            {
                "task": "class9_classification",
                "subset": "class9_inference2_gold",
                "metric": "macro_f1",
                "value": metrics["macro_f1"],
                "role": "PAPER_CORE",
            },
        ]
    ).to_csv(
        PAPER_CORE_CSV,
        index=False,
    )


# =============================================================================
# 7. Main
# =============================================================================

def main():
    logger = build_logger()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("=" * 78)
    logger.info(
        "STAGE-1 CLASS-9 MANUAL GOLD CLASSIFICATION CHALLENGE"
    )
    logger.info("=" * 78)
    logger.info("Model          : %s", MODEL_NAME)
    logger.info("Device         : %s", DEVICE)
    logger.info("Gold CSV       : %s", GOLD_CSV)

    gold_df = load_and_validate_gold()

    label_counts_all = (
        gold_df["manual_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    included_df = gold_df[
        gold_df["_included_classification"]
    ]

    logger.info("Gold rows       : %d", len(gold_df))
    logger.info("Manual labels   : %s", label_counts_all)
    logger.info("Evaluated rows  : %d", len(included_df))
    logger.info(
        "Excluded (-1)   : %d",
        int((gold_df["manual_label"] == -1).sum()),
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Stage-1 best checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != MODEL_NAME:
        raise ValueError(
            "Checkpoint model mismatch: "
            f"{checkpoint_model_name!r} != {MODEL_NAME!r}"
        )

    checkpoint_num_classes = int(
        checkpoint.get(
            "num_classes",
            NUM_CLASSES,
        )
    )
    if checkpoint_num_classes != NUM_CLASSES:
        raise ValueError(
            "Checkpoint num_classes mismatch: "
            f"{checkpoint_num_classes} != {NUM_CLASSES}"
        )

    normalization = checkpoint.get("normalization")
    if not normalization:
        raise KeyError(
            "Checkpoint has no normalization information."
        )

    mean = float(normalization["mean"])
    std = float(normalization["std"])

    logger.info(
        "Best epoch      : %s",
        checkpoint.get("best_epoch"),
    )
    logger.info(
        "Best val MacroF1: %.6f",
        float(
            checkpoint.get(
                "best_val_macro_f1",
                np.nan,
            )
        ),
    )
    logger.info(
        "Normalization   : mean=%.8f, std=%.8f",
        mean,
        std,
    )

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
            "Dataset row count != Gold CSV row count."
        )

    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = stage1_module.build_model(
        MODEL_NAME,
        NUM_CLASSES,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        records,
        feature_records,
    ) = run_inference(
        model=model,
        loader=loader,
        gold_df=gold_df,
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

    pd.DataFrame(records).to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    evaluated = [
        r
        for r in records
        if r["included_classification"]
    ]

    y_true = np.asarray(
        [r["manual_label"] for r in evaluated],
        dtype=int,
    )
    y_pred = np.asarray(
        [r["pred_class"] for r in evaluated],
        dtype=int,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    save_full_metrics(metrics)
    save_paper_core(metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        REPORT_CSV
    )

    cm = np.asarray(
        metrics["confusion_matrix"],
        dtype=int,
    )
    pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    ).to_csv(CONFUSION_CSV)

    logger.info("=" * 78)
    logger.info("STAGE-1 CLASS-9 GOLD RESULTS")
    logger.info("=" * 78)
    logger.info(
        "[PAPER CORE] Accuracy          : %.6f",
        metrics["accuracy"],
    )
    logger.info(
        "[PAPER CORE] Balanced Accuracy : %.6f",
        metrics["balanced_accuracy"],
    )
    logger.info(
        "[PAPER CORE] Macro F1          : %.6f",
        metrics["macro_f1"],
    )

    logger.info("-" * 78)
    logger.info(
        "[AUX] Macro Precision          : %.6f",
        metrics["macro_precision"],
    )
    logger.info(
        "[AUX] Macro Recall             : %.6f",
        metrics["macro_recall"],
    )
    logger.info(
        "[AUX] Weighted F1              : %.6f",
        metrics["weighted_f1"],
    )

    logger.info("-" * 78)
    for label, name in enumerate(CLASS_NAMES):
        m = metrics["per_class"][str(label)]
        logger.info(
            "[AUX] class-%d %-18s | P=%.6f R=%.6f F1=%.6f N=%d",
            label,
            name,
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
        )

    logger.info(
        "Output directory             : %s",
        OUTPUT_DIR,
    )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
