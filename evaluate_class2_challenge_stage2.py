# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_class2_challenge.py
============================

Dark Soliton 项目：class-2 人工 Gold Challenge 独立泛化评价脚本。

"""

from pathlib import Path
from functools import lru_cache
import ast
import json
import logging
import sys

import numpy as np
import pandas as pd

import torch

import train_detector_baseline as detector_module

from dark_soliton_dataloader import (
    DarkSolitonDataset,
    make_dataloader,
)


# ======================================================================
# 1. Configuration
# ======================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MANUAL_ANNOTATION_CSV = (
    DATA_DIR
    / "class2_manual_annotations.csv"
)

# MODEL_NAME = "simple_cnn"
MODEL_NAME = "resnet18"
# MODEL_NAME = "repvit_m0_9"
# MODEL_NAME = "fasternet_t0"

# seed_number = "seed42"
# seed_number = "seed90"
seed_number = "seed2027"
# 若同一个 CSV 中存在多个 annotator：
# - 指定 "A" / "B" 可单独评价某位标注者；
# - 若已经生成 consensus Gold，可设为 None。
ANNOTATOR = "A"

BATCH_SIZE = 32
NUM_WORKERS = 2

PRIMARY_TOLERANCE_PX = 2.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_OUTPUT_ROOT = (
    ROOT
    / "output"
    / seed_number
    / MODEL_NAME
    / "detector_baseline"
)

CHECKPOINT_PATH = (
    MODEL_OUTPUT_ROOT
    / "checkpoint"
    / "best.pt.tar"
)

VAL_THRESHOLD_CSV = (
    MODEL_OUTPUT_ROOT
    / "metrics"
    / "val_threshold_search.csv"
)

CHALLENGE_OUTPUT_DIR = (
    MODEL_OUTPUT_ROOT
    / "gold_challenge_combined"
)

PREDICTIONS_CSV = (
    CHALLENGE_OUTPUT_DIR
    / "challenge_predictions.csv"
)

METRICS_JSON = (
    CHALLENGE_OUTPUT_DIR
    / "challenge_metrics.json"
)

PER_COUNT_CSV = (
    CHALLENGE_OUTPUT_DIR
    / "per_count_metrics.csv"
)

ANNOTATION_SUMMARY_JSON = (
    CHALLENGE_OUTPUT_DIR
    / "annotation_summary.json"
)

# Paper-oriented outputs, aligned with Stage 3 / Stage 4.
CLASS2_LOCALIZATION_JSON = (
    CHALLENGE_OUTPUT_DIR
    / "class2_localization_metrics.json"
)

PAPER_CORE_JSON = (
    CHALLENGE_OUTPUT_DIR
    / "paper_core_metrics.json"
)

PAPER_CORE_CSV = (
    CHALLENGE_OUTPUT_DIR
    / "paper_core_metrics.csv"
)

# Deepest-backbone Gold features for t-SNE / UMAP.
GOLD_FEATURES_CSV = (
    CHALLENGE_OUTPUT_DIR
    / "gold_features.csv"
)


# ======================================================================
# 2. Logging
# ======================================================================

def build_logger():
    logger = logging.getLogger(
        "gold_challenge_combined"
    )

    logger.setLevel(
        logging.INFO
    )
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(
        sys.stdout
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )

    return logger


# ======================================================================
# 3. Annotation parsing / validation
# ======================================================================

def parse_position_list(value):
    """
    支持：
        [48.3, 103.6]
    的 JSON / Python-list 字符串形式。
    """
    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and np.isnan(value):
        return []

    if isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        raw = list(value)

    else:
        text = str(
            value
        ).strip()

        if not text:
            return []

        try:
            raw = json.loads(
                text
            )
        except Exception:
            raw = ast.literal_eval(
                text
            )

    if not isinstance(
        raw,
        (list, tuple),
    ):
        raise ValueError(
            f"excitation_positions 不是 list: {value!r}"
        )

    positions = [
        float(x)
        for x in raw
    ]

    positions = sorted(
        positions
    )

    return positions


def load_and_validate_annotations():
    if not MANUAL_ANNOTATION_CSV.exists():
        raise FileNotFoundError(
            f"人工标注 CSV 不存在：{MANUAL_ANNOTATION_CSV}"
        )

    df = pd.read_csv(
        MANUAL_ANNOTATION_CSV
    )

    required_columns = {
        "file_name",
        "relative_path",
        "num_excitations",
        "excitation_positions",
        "annotation_status",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            "人工标注 CSV 缺少字段："
            f"{sorted(missing)}"
        )

    if (
        ANNOTATOR is not None
        and "annotator" in df.columns
    ):
        df = df[
            df["annotator"].astype(
                str
            ) == str(
                ANNOTATOR
            )
        ].copy()

        if len(df) == 0:
            raise RuntimeError(
                f"找不到 annotator={ANNOTATOR!r} 的标注。"
            )

    df = df.reset_index(
        drop=True
    )

    duplicate_names = (
        df["file_name"]
        .astype(str)
        .duplicated(
            keep=False
        )
    )

    if duplicate_names.any():
        examples = (
            df.loc[
                duplicate_names,
                "file_name",
            ]
            .astype(str)
            .tolist()[:10]
        )

        raise ValueError(
            "人工标注表中同一 file_name 出现多行。"
            "请先生成单一 annotator 或 consensus Gold 表。"
            f"示例：{examples}"
        )

    parsed_positions = []

    for row_index, row in df.iterrows():
        status = str(
            row[
                "annotation_status"
            ]
        ).strip()

        positions = parse_position_list(
            row[
                "excitation_positions"
            ]
        )

        num_excitations = int(
            row[
                "num_excitations"
            ]
        )

        for position in positions:
            if not (
                0.0
                <= position
                <= 163.0
            ):
                raise ValueError(
                    f"{row['file_name']}: "
                    f"position={position} 超出 [0,163]"
                )

        if status == "valid":
            if (
                num_excitations
                != len(
                    positions
                )
            ):
                raise ValueError(
                    f"{row['file_name']}: "
                    "valid 样本的 num_excitations "
                    "与 excitation_positions 数量不一致。"
                )

            if num_excitations <= 0:
                raise ValueError(
                    f"{row['file_name']}: "
                    "valid 样本应至少有一个可靠位置。"
                )

        elif status == "no_reliable_soliton":
            if (
                num_excitations != 0
                or len(
                    positions
                ) != 0
            ):
                raise ValueError(
                    f"{row['file_name']}: "
                    "no_reliable_soliton 必须为 0 个 GT。"
                )

        elif status == "ambiguous":
            # ambiguous 不进入 primary metrics，
            # 因此不强行解释其 0 个位置为 negative GT。
            pass

        else:
            raise ValueError(
                f"{row['file_name']}: "
                f"未知 annotation_status={status!r}"
            )

        parsed_positions.append(
            positions
        )

    df[
        "_gt_positions"
    ] = parsed_positions

    return df


# ======================================================================
# 4. Threshold from VALIDATION only
# ======================================================================

def select_validation_threshold():
    if not VAL_THRESHOLD_CSV.exists():
        raise FileNotFoundError(
            "找不到 Stage 2 validation threshold 文件："
            f"{VAL_THRESHOLD_CSV}"
        )

    df = pd.read_csv(
        VAL_THRESHOLD_CSV
    )

    required = {
        "threshold",
        "precision_at_2px",
        "f1_at_2px",
        "false_positives_per_image",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            "val_threshold_search.csv 缺少字段："
            f"{sorted(missing)}"
        )

    ranked = df.sort_values(
        by=[
            "f1_at_2px",
            "precision_at_2px",
            "false_positives_per_image",
            "threshold",
        ],
        ascending=[
            False,
            False,
            True,
            False,
        ],
        kind="mergesort",
    )

    best = ranked.iloc[
        0
    ]

    return (
        float(
            best[
                "threshold"
            ]
        ),
        best.to_dict(),
    )


# ======================================================================
# 5. One-to-one matching
# ======================================================================

def match_positions(
    gt_positions,
    pred_positions,
    tolerance_px,
):
    """
    一维 one-to-one optimal matching。

    第一目标：
        最大化匹配数量。

    第二目标：
        在匹配数相同情况下最小化总误差。

    返回：
        list of dict:
        {
            "gt": ...,
            "pred": ...,
            "error": ...
        }
    """
    gt = tuple(
        sorted(
            float(x)
            for x in gt_positions
        )
    )

    pred = tuple(
        sorted(
            float(x)
            for x in pred_positions
        )
    )

    @lru_cache(
        maxsize=None
    )
    def solve(i, j):
        if (
            i >= len(gt)
            or j >= len(pred)
        ):
            return (
                0,
                0.0,
                (),
            )

        candidates = []

        # Skip GT
        match_count, error_sum, pairs = (
            solve(
                i + 1,
                j,
            )
        )

        candidates.append((
            match_count,
            error_sum,
            pairs,
        ))

        # Skip prediction
        match_count, error_sum, pairs = (
            solve(
                i,
                j + 1,
            )
        )

        candidates.append((
            match_count,
            error_sum,
            pairs,
        ))

        # Match
        error = abs(
            gt[i]
            - pred[j]
        )

        if (
            error
            <= tolerance_px
        ):
            (
                sub_count,
                sub_error,
                sub_pairs,
            ) = solve(
                i + 1,
                j + 1,
            )

            candidates.append((
                sub_count + 1,
                sub_error + error,
                (
                    (
                        gt[i],
                        pred[j],
                        error,
                    ),
                )
                + sub_pairs,
            ))

        # max match count, then min error
        best = sorted(
            candidates,
            key=lambda x: (
                -x[0],
                x[1],
            ),
        )[0]

        return best

    (
        match_count,
        _,
        pairs,
    ) = solve(
        0,
        0,
    )

    assert (
        match_count
        == len(
            pairs
        )
    )

    return [
        {
            "gt": float(
                gt_position
            ),
            "pred": float(
                pred_position
            ),
            "error": float(
                error
            ),
        }
        for (
            gt_position,
            pred_position,
            error,
        ) in pairs
    ]


# ======================================================================
# 6. Metrics
# ======================================================================

def calculate_challenge_metrics(
    records,
    tolerance_px,
):
    """
    仅使用 included_primary=True 的样本。
    """
    included = [
        record
        for record in records
        if record[
            "included_primary"
        ]
    ]

    if not included:
        raise RuntimeError(
            "没有可用于 primary metrics 的样本。"
        )

    total_gt = 0
    total_pred = 0
    total_tp = 0

    all_matched_errors = []

    exact_count_hits = 0
    exact_sample_hits = 0
    count_abs_errors = []

    for record in included:
        gt_positions = record[
            "gt_positions"
        ]

        pred_positions = record[
            "pred_positions"
        ]

        pairs = match_positions(
            gt_positions,
            pred_positions,
            tolerance_px,
        )

        n_gt = len(
            gt_positions
        )

        n_pred = len(
            pred_positions
        )

        n_tp = len(
            pairs
        )

        total_gt += n_gt
        total_pred += n_pred
        total_tp += n_tp

        all_matched_errors.extend(
            pair[
                "error"
            ]
            for pair in pairs
        )

        exact_count = (
            n_pred
            == n_gt
        )

        exact_sample = (
            exact_count
            and n_tp
            == n_gt
        )

        exact_count_hits += int(
            exact_count
        )

        exact_sample_hits += int(
            exact_sample
        )

        count_abs_errors.append(
            abs(
                n_pred
                - n_gt
            )
        )

    fp = (
        total_pred
        - total_tp
    )

    fn = (
        total_gt
        - total_tp
    )

    precision = (
        total_tp
        / (
            total_tp
            + fp
        )
        if (
            total_tp
            + fp
        ) > 0
        else 0.0
    )

    recall = (
        total_tp
        / (
            total_tp
            + fn
        )
        if (
            total_tp
            + fn
        ) > 0
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
                all_matched_errors
            )
        )
        if all_matched_errors
        else float(
            "nan"
        )
    )

    return {
        "num_images": int(
            len(
                included
            )
        ),
        "total_gt_excitations": int(
            total_gt
        ),
        "total_predictions": int(
            total_pred
        ),

        "true_positives": int(
            total_tp
        ),
        "false_positives": int(
            fp
        ),
        "false_negatives": int(
            fn
        ),

        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),

        "matched_mae_px": float(
            matched_mae
        ),

        "exact_count_accuracy": float(
            exact_count_hits
            / len(
                included
            )
        ),

        "count_mae": float(
            np.mean(
                count_abs_errors
            )
        ),

        "exact_sample_success": float(
            exact_sample_hits
            / len(
                included
            )
        ),

        "false_positives_per_image": float(
            fp
            / len(
                included
            )
        ),

        "false_negatives_per_image": float(
            fn
            / len(
                included
            )
        ),
    }


def calculate_gt_match_rate(
    records,
    tolerance_px,
):
    """
    多目标场景下的 GT Match Rate：
        matched GT / total GT

    本质上对应给定容差下的位置 Recall，
    但单独报告有助于与 Acc@1/2/3 的直觉对应。
    """
    included = [
        record
        for record in records
        if record[
            "included_primary"
        ]
    ]

    total_gt = 0
    total_matched = 0

    for record in included:
        gt_positions = record[
            "gt_positions"
        ]

        pred_positions = record[
            "pred_positions"
        ]

        pairs = match_positions(
            gt_positions,
            pred_positions,
            tolerance_px,
        )

        total_gt += len(
            gt_positions
        )

        total_matched += len(
            pairs
        )

    if total_gt == 0:
        return float(
            "nan"
        )

    return float(
        total_matched
        / total_gt
    )


# ======================================================================
# 7. Per-count summary
# ======================================================================

def build_per_count_metrics(
    records,
):
    rows = []

    primary_records = [
        record
        for record in records
        if record[
            "included_primary"
        ]
    ]

    gt_counts = sorted(
        set(
            len(
                record[
                    "gt_positions"
                ]
            )
            for record in primary_records
        )
    )

    for gt_count in gt_counts:
        subset = [
            record
            for record in primary_records
            if len(
                record[
                    "gt_positions"
                ]
            )
            == gt_count
        ]

        if not subset:
            continue

        metrics = calculate_challenge_metrics(
            subset,
            tolerance_px=PRIMARY_TOLERANCE_PX,
        )

        rows.append({
            "gt_count": int(
                gt_count
            ),
            "num_images": int(
                len(
                    subset
                )
            ),
            "precision_at_2px": (
                metrics[
                    "precision"
                ]
            ),
            "recall_at_2px": (
                metrics[
                    "recall"
                ]
            ),
            "f1_at_2px": (
                metrics[
                    "f1"
                ]
            ),
            "matched_mae_px": (
                metrics[
                    "matched_mae_px"
                ]
            ),
            "exact_count_accuracy": (
                metrics[
                    "exact_count_accuracy"
                ]
            ),
            "count_mae": (
                metrics[
                    "count_mae"
                ]
            ),
            "exact_sample_success_at_2px": (
                metrics[
                    "exact_sample_success"
                ]
            ),
        })

    return pd.DataFrame(
        rows
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


# ======================================================================
# 8. Main evaluation
# ======================================================================

@torch.no_grad()
def run_challenge_evaluation(
    model,
    loader,
    annotation_df,
    threshold,
):
    model.eval()

    records = []
    feature_records = []

    for batch in loader:
        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=(
                DEVICE.type
                == "cuda"
            ),
        )

        # Original:
        # outputs = model(images)
        #
        # Split the exact same detector forward into backbone + head so the
        # deepest backbone representation can be exported without a second
        # network pass.
        backbone_features = (
            model.backbone(
                images
            )
        )

        outputs = model.head(
            backbone_features
        )

        shared_features = (
            backbone_features[-1]
            .mean(dim=(2, 3))
        )

        heatmap_logits = outputs[
            "heatmap_logits"
        ]

        offset_logits = outputs[
            "offset_logits"
        ]

        indices = batch[
            "index"
        ].cpu().tolist()

        for batch_index, row_index in enumerate(
            indices
        ):
            row = annotation_df.iloc[
                int(
                    row_index
                )
            ]

            decoded = (
                detector_module
                .decode_single_prediction(
                    heatmap_logits=heatmap_logits[
                        batch_index
                    ],
                    offset_logits=offset_logits[
                        batch_index
                    ],
                    threshold=threshold,
                )
            )

            status = str(
                row[
                    "annotation_status"
                ]
            ).strip()

            included_primary = (
                status
                in {
                    "valid",
                    "no_reliable_soliton",
                }
            )

            gt_positions = (
                list(
                    row[
                        "_gt_positions"
                    ]
                )
                if included_primary
                else []
            )

            records.append({
                "file_name": str(
                    row[
                        "file_name"
                    ]
                ),
                "annotation_status": (
                    status
                ),
                "confidence": str(
                    row.get(
                        "confidence",
                        "",
                    )
                ),
                "included_primary": bool(
                    included_primary
                ),
                "gt_positions": (
                    gt_positions
                ),
                "pred_positions": (
                    decoded[
                        "pred_positions"
                    ]
                ),
                "pred_scores": (
                    decoded[
                        "pred_scores"
                    ]
                ),
            })

            feature_vector = (
                shared_features[
                    batch_index
                ]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            feature_row = {
                "file_name": str(
                    row["file_name"]
                ),
                "annotation_status": status,
                "confidence": str(
                    row.get(
                        "confidence",
                        "",
                    )
                ),
                "included_primary": bool(
                    included_primary
                ),
                "num_excitations": int(
                    row["num_excitations"]
                ),
                "feature_type": (
                    "deepest_backbone_global_average_pool"
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


# ======================================================================
# 9. Save detailed predictions
# ======================================================================

def save_prediction_table(
    records,
):
    rows = []

    for record in records:
        gt_positions = record[
            "gt_positions"
        ]

        pred_positions = record[
            "pred_positions"
        ]

        if record[
            "included_primary"
        ]:
            pairs_at_2 = match_positions(
                gt_positions,
                pred_positions,
                PRIMARY_TOLERANCE_PX,
            )

            tp_at_2 = len(
                pairs_at_2
            )

            fp_at_2 = (
                len(
                    pred_positions
                )
                - tp_at_2
            )

            fn_at_2 = (
                len(
                    gt_positions
                )
                - tp_at_2
            )

            exact_count = (
                len(
                    pred_positions
                )
                == len(
                    gt_positions
                )
            )

            exact_sample = (
                exact_count
                and tp_at_2
                == len(
                    gt_positions
                )
            )

        else:
            pairs_at_2 = []
            tp_at_2 = np.nan
            fp_at_2 = np.nan
            fn_at_2 = np.nan
            exact_count = np.nan
            exact_sample = np.nan

        rows.append({
            "file_name": record[
                "file_name"
            ],
            "annotation_status": (
                record[
                    "annotation_status"
                ]
            ),
            "confidence": (
                record[
                    "confidence"
                ]
            ),
            "included_primary": (
                record[
                    "included_primary"
                ]
            ),

            "gt_count": (
                len(
                    gt_positions
                )
                if record[
                    "included_primary"
                ]
                else np.nan
            ),

            "gt_positions": json.dumps(
                [
                    round(
                        float(x),
                        6,
                    )
                    for x in gt_positions
                ]
            ),

            "pred_count": len(
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
                    for x in record[
                        "pred_scores"
                    ]
                ]
            ),

            "matched_pairs_at_2px": (
                json.dumps(
                    [
                        {
                            "gt": round(
                                pair[
                                    "gt"
                                ],
                                6,
                            ),
                            "pred": round(
                                pair[
                                    "pred"
                                ],
                                6,
                            ),
                            "error": round(
                                pair[
                                    "error"
                                ],
                                6,
                            ),
                        }
                        for pair in pairs_at_2
                    ]
                )
                if record[
                    "included_primary"
                ]
                else ""
            ),

            "tp_at_2px": (
                tp_at_2
            ),
            "fp_at_2px": (
                fp_at_2
            ),
            "fn_at_2px": (
                fn_at_2
            ),

            "exact_count": (
                exact_count
            ),

            "exact_sample_success_at_2px": (
                exact_sample
            ),
        })

    pd.DataFrame(
        rows
    ).to_csv(
        PREDICTIONS_CSV,
        index=False,
    )


# ======================================================================
# 10. Main
# ======================================================================

def main():
    logger = build_logger()

    CHALLENGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "=" * 78
    )
    logger.info(
        "Class-2 Manual Gold Challenge Evaluation"
    )
    logger.info(
        "=" * 78
    )

    logger.info(
        "Model          : %s",
        MODEL_NAME,
    )

    logger.info(
        "Device         : %s",
        DEVICE,
    )

    annotation_df = (
        load_and_validate_annotations()
    )

    status_counts = (
        annotation_df[
            "annotation_status"
        ]
        .value_counts()
        .to_dict()
    )

    valid_df = annotation_df[
        annotation_df[
            "annotation_status"
        ]
        == "valid"
    ]

    valid_count_distribution = (
        valid_df[
            "num_excitations"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    logger.info(
        "Annotations    : N=%d",
        len(
            annotation_df
        ),
    )

    logger.info(
        "Status counts  : %s",
        status_counts,
    )

    logger.info(
        "Valid GT-count : %s",
        valid_count_distribution,
    )

    annotation_summary = {
        "total_rows": int(
            len(
                annotation_df
            )
        ),
        "status_counts": {
            str(k): int(v)
            for k, v in (
                status_counts.items()
            )
        },
        "valid_gt_count_distribution": {
            str(k): int(v)
            for k, v in (
                valid_count_distribution.items()
            )
        },
        "primary_included_statuses": [
            "valid",
            "no_reliable_soliton",
        ],
        "excluded_primary_statuses": [
            "ambiguous",
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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint 不存在：{CHECKPOINT_PATH}"
        )

    selected_threshold, threshold_row = (
        select_validation_threshold()
    )

    logger.info(
        "Threshold      : %.2f "
        "(selected from Stage-2 validation only)",
        selected_threshold,
    )

    logger.info(
        "Val F1@2       : %.6f",
        float(
            threshold_row[
                "f1_at_2px"
            ]
        ),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    checkpoint_model_name = checkpoint.get(
        "model_name"
    )

    if (
        checkpoint_model_name
        != MODEL_NAME
    ):
        raise ValueError(
            "Checkpoint model mismatch: "
            f"{checkpoint_model_name!r} != {MODEL_NAME!r}"
        )

    checkpoint_num_bins = checkpoint.get(
        "num_bins"
    )

    if (
        checkpoint_num_bins is not None
        and int(
            checkpoint_num_bins
        )
        != int(
            detector_module.NUM_BINS
        )
    ):
        raise ValueError(
            "Checkpoint num_bins 与当前 "
            "train_detector_baseline.py 不一致。"
        )

    normalization = checkpoint.get(
        "normalization"
    )

    if not normalization:
        raise KeyError(
            "Checkpoint 缺少 normalization 信息。"
        )

    mean = float(
        normalization[
            "mean"
        ]
    )

    std = float(
        normalization[
            "std"
        ]
    )

    dataset = DarkSolitonDataset(
        csv_path=MANUAL_ANNOTATION_CSV,
        task="unlabeled",
        normalize="dataset_zscore",
        dataset_mean=mean,
        dataset_std=std,
        check_file_exists=True,
    )

    # 如果 CSV 中包含多个 annotator，
    # DarkSolitonDataset 会读取原 CSV 全部行；
    # 因此本脚本要求当前人工标注表已经是单 annotator/consensus。
    # 当前 ANNOTATOR 过滤后的 df 数量必须与 Dataset 一致。
    if len(
        dataset
    ) != len(
        annotation_df
    ):
        raise RuntimeError(
            "人工标注 CSV 过滤后行数与 Dataset 不一致。"
            "如果同一文件包含多个 annotator，"
            "建议先导出单独的 annotator/consensus CSV。"
        )

    loader = make_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type
            == "cuda"
        ),
    )

    model = (
        detector_module
        .DarkSolitonDetector(
            MODEL_NAME
        )
        .to(
            DEVICE
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        records,
        feature_records,
    ) = run_challenge_evaluation(
        model=model,
        loader=loader,
        annotation_df=annotation_df,
        threshold=selected_threshold,
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

    save_prediction_table(
        records
    )

    primary_metrics = (
        calculate_challenge_metrics(
            records,
            tolerance_px=(
                PRIMARY_TOLERANCE_PX
            ),
        )
    )

    gt_match_rate_1 = (
        calculate_gt_match_rate(
            records,
            tolerance_px=1.0,
        )
    )

    gt_match_rate_2 = (
        calculate_gt_match_rate(
            records,
            tolerance_px=2.0,
        )
    )

    gt_match_rate_3 = (
        calculate_gt_match_rate(
            records,
            tolerance_px=3.0,
        )
    )

    output_metrics = {
        "model_name": MODEL_NAME,
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "threshold": float(
            selected_threshold
        ),
        "threshold_source": str(
            VAL_THRESHOLD_CSV
        ),

        "primary_tolerance_px": (
            PRIMARY_TOLERANCE_PX
        ),

        "included_statuses": [
            "valid",
            "no_reliable_soliton",
        ],

        "excluded_statuses": [
            "ambiguous",
        ],

        "precision_at_2px": (
            primary_metrics[
                "precision"
            ]
        ),

        "recall_at_2px": (
            primary_metrics[
                "recall"
            ]
        ),

        "f1_at_2px": (
            primary_metrics[
                "f1"
            ]
        ),

        "gt_match_rate_at_1px": (
            gt_match_rate_1
        ),

        "gt_match_rate_at_2px": (
            gt_match_rate_2
        ),

        "gt_match_rate_at_3px": (
            gt_match_rate_3
        ),

        "matched_mae_at_2px": (
            primary_metrics[
                "matched_mae_px"
            ]
        ),

        "exact_count_accuracy": (
            primary_metrics[
                "exact_count_accuracy"
            ]
        ),

        "count_mae": (
            primary_metrics[
                "count_mae"
            ]
        ),

        "exact_sample_success_at_2px": (
            primary_metrics[
                "exact_sample_success"
            ]
        ),

        "false_positives_per_image": (
            primary_metrics[
                "false_positives_per_image"
            ]
        ),

        "false_negatives_per_image": (
            primary_metrics[
                "false_negatives_per_image"
            ]
        ),

        "num_primary_images": (
            primary_metrics[
                "num_images"
            ]
        ),

        "total_gt_excitations": (
            primary_metrics[
                "total_gt_excitations"
            ]
        ),

        "total_predictions": (
            primary_metrics[
                "total_predictions"
            ]
        ),

        "true_positives": (
            primary_metrics[
                "true_positives"
            ]
        ),

        "false_positives": (
            primary_metrics[
                "false_positives"
            ]
        ),

        "false_negatives": (
            primary_metrics[
                "false_negatives"
            ]
        ),
    }

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_metrics,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    # -----------------------------------------------------------------
    # Paper-oriented Stage-2 class-2 localization outputs.
    # Keep challenge_metrics.json above for backward compatibility.
    # Metric names below intentionally match Stage 3 / Stage 4.
    # -----------------------------------------------------------------
    class2_loc_metrics = {
        "num_images": int(primary_metrics["num_images"]),
        "total_gt_excitations": int(
            primary_metrics["total_gt_excitations"]
        ),
        "total_predictions": int(
            primary_metrics["total_predictions"]
        ),
        "true_positives": int(
            primary_metrics["true_positives"]
        ),
        "false_positives": int(
            primary_metrics["false_positives"]
        ),
        "false_negatives": int(
            primary_metrics["false_negatives"]
        ),
        "precision_at_2px": float(
            primary_metrics["precision"]
        ),
        "recall_at_2px": float(
            primary_metrics["recall"]
        ),
        "f1_at_2px": float(
            primary_metrics["f1"]
        ),
        "matched_mae_at_2px": float(
            primary_metrics["matched_mae_px"]
        ),
        "exact_count_accuracy": float(
            primary_metrics["exact_count_accuracy"]
        ),
        "count_mae": float(
            primary_metrics["count_mae"]
        ),
        "exact_sample_success_at_2px": float(
            primary_metrics["exact_sample_success"]
        ),
        "false_positives_per_image": float(
            primary_metrics["false_positives_per_image"]
        ),
        "false_negatives_per_image": float(
            primary_metrics["false_negatives_per_image"]
        ),
        "gt_match_rate_at_1px": float(
            gt_match_rate_1
        ),
        "gt_match_rate_at_2px": float(
            gt_match_rate_2
        ),
        "gt_match_rate_at_3px": float(
            gt_match_rate_3
        ),
    }

    with open(
        CLASS2_LOCALIZATION_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class2_loc_metrics,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    paper_core = {
        "stage": "stage2_detector",
        "model_name": MODEL_NAME,
        "gold_csv": str(
            MANUAL_ANNOTATION_CSV
        ),
        "paper_primary_tasks": {
            "class2_localization": {
                "subset": "class2_test_gold",
                "eligible_images": (
                    class2_loc_metrics["num_images"]
                ),
                "precision_at_2px": (
                    class2_loc_metrics["precision_at_2px"]
                ),
                "recall_at_2px": (
                    class2_loc_metrics["recall_at_2px"]
                ),
                "f1_at_2px": (
                    class2_loc_metrics["f1_at_2px"]
                ),
                "matched_mae_at_2px": (
                    class2_loc_metrics["matched_mae_at_2px"]
                ),
                "exact_count_accuracy": (
                    class2_loc_metrics["exact_count_accuracy"]
                ),
                "exact_sample_success_at_2px": (
                    class2_loc_metrics[
                        "exact_sample_success_at_2px"
                    ]
                ),
            },
        },
        "note": (
            "Recommended paper-primary comparison: "
            "Stage2/Stage3/Stage4 on class-2 localization. "
            "Gold annotations are used only for final evaluation "
            "and never for training, validation, checkpoint selection, "
            "or threshold tuning."
        ),
    }

    with open(
        PAPER_CORE_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            paper_core,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    paper_rows = []

    for metric_name in [
        "precision_at_2px",
        "recall_at_2px",
        "f1_at_2px",
        "matched_mae_at_2px",
        "exact_count_accuracy",
        "exact_sample_success_at_2px",
    ]:
        paper_rows.append({
            "stage": "stage2",
            "task": "localization",
            "subset": "class2_test_gold",
            "metric": metric_name,
            "value": class2_loc_metrics[
                metric_name
            ],
            "role": "PAPER_CORE",
        })

    pd.DataFrame(
        paper_rows
    ).to_csv(
        PAPER_CORE_CSV,
        index=False,
    )

    per_count_df = (
        build_per_count_metrics(
            records
        )
    )

    per_count_df.to_csv(
        PER_COUNT_CSV,
        index=False,
    )

    logger.info(
        "=" * 78
    )

    logger.info(
        "CLASS-2 GENERALIZATION RESULTS"
    )

    logger.info(
        "=" * 78
    )

    logger.info(
        "Primary images             : %d",
        primary_metrics[
            "num_images"
        ],
    )

    logger.info(
        "Total GT excitations       : %d",
        primary_metrics[
            "total_gt_excitations"
        ],
    )

    logger.info(
        "Precision @ 2 px           : %.6f",
        primary_metrics[
            "precision"
        ],
    )

    logger.info(
        "Recall @ 2 px              : %.6f",
        primary_metrics[
            "recall"
        ],
    )

    logger.info(
        "F1 @ 2 px                  : %.6f",
        primary_metrics[
            "f1"
        ],
    )

    logger.info(
        "GT Match Rate @ 1 px       : %.6f",
        gt_match_rate_1,
    )

    logger.info(
        "GT Match Rate @ 2 px       : %.6f",
        gt_match_rate_2,
    )

    logger.info(
        "GT Match Rate @ 3 px       : %.6f",
        gt_match_rate_3,
    )

    logger.info(
        "Matched MAE @ 2 px         : %.6f px",
        primary_metrics[
            "matched_mae_px"
        ],
    )

    logger.info(
        "Exact Count Accuracy       : %.6f",
        primary_metrics[
            "exact_count_accuracy"
        ],
    )

    logger.info(
        "Count MAE                  : %.6f",
        primary_metrics[
            "count_mae"
        ],
    )

    logger.info(
        "Exact Sample Success @ 2px : %.6f",
        primary_metrics[
            "exact_sample_success"
        ],
    )

    logger.info(
        "False positives / image    : %.6f",
        primary_metrics[
            "false_positives_per_image"
        ],
    )

    logger.info(
        "False negatives / image    : %.6f",
        primary_metrics[
            "false_negatives_per_image"
        ],
    )

    logger.info("=" * 78)
    logger.info(
        "STAGE-2 CLASS-2 GOLD LOCALIZATION — PAPER PRIMARY"
    )
    logger.info("=" * 78)

    logger.info(
        "[PAPER CORE] Precision @ 2 px           : %.6f",
        class2_loc_metrics["precision_at_2px"],
    )
    logger.info(
        "[PAPER CORE] Recall @ 2 px              : %.6f",
        class2_loc_metrics["recall_at_2px"],
    )
    logger.info(
        "[PAPER CORE] F1 @ 2 px                  : %.6f",
        class2_loc_metrics["f1_at_2px"],
    )
    logger.info(
        "[PAPER CORE] Matched MAE @ 2 px         : %.6f px",
        class2_loc_metrics["matched_mae_at_2px"],
    )
    logger.info(
        "[PAPER CORE] Exact Count Accuracy       : %.6f",
        class2_loc_metrics["exact_count_accuracy"],
    )
    logger.info(
        "[PAPER CORE] Exact Sample Success @ 2px : %.6f",
        class2_loc_metrics[
            "exact_sample_success_at_2px"
        ],
    )

    logger.info("-" * 78)
    logger.info(
        "[AUX] Eligible images        : %d",
        class2_loc_metrics["num_images"],
    )
    logger.info(
        "[AUX] Total GT excitations   : %d",
        class2_loc_metrics["total_gt_excitations"],
    )
    logger.info(
        "[AUX] Total predictions      : %d",
        class2_loc_metrics["total_predictions"],
    )
    logger.info(
        "[AUX] TP / FP / FN           : %d / %d / %d",
        class2_loc_metrics["true_positives"],
        class2_loc_metrics["false_positives"],
        class2_loc_metrics["false_negatives"],
    )
    logger.info(
        "[AUX] GT Match Rate @ 1 px   : %.6f",
        class2_loc_metrics["gt_match_rate_at_1px"],
    )
    logger.info(
        "[AUX] GT Match Rate @ 2 px   : %.6f",
        class2_loc_metrics["gt_match_rate_at_2px"],
    )
    logger.info(
        "[AUX] GT Match Rate @ 3 px   : %.6f",
        class2_loc_metrics["gt_match_rate_at_3px"],
    )
    logger.info(
        "[AUX] Count MAE              : %.6f",
        class2_loc_metrics["count_mae"],
    )
    logger.info(
        "[AUX] FP / image             : %.6f",
        class2_loc_metrics["false_positives_per_image"],
    )
    logger.info(
        "[AUX] FN / image             : %.6f",
        class2_loc_metrics["false_negatives_per_image"],
    )

    logger.info("=" * 78)
    logger.info("PAPER-CORE OUTPUTS")
    logger.info("=" * 78)
    logger.info(
        "Core JSON                    : %s",
        PAPER_CORE_JSON,
    )
    logger.info(
        "Core CSV                     : %s",
        PAPER_CORE_CSV,
    )
    logger.info(
        "Class2 localization JSON     : %s",
        CLASS2_LOCALIZATION_JSON,
    )
    logger.info(
        "Gold features                : %s",
        GOLD_FEATURES_CSV,
    )
    logger.info(
        "Output directory             : %s",
        CHALLENGE_OUTPUT_DIR,
    )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
