# # 运行完prepare_dataset.py后，生成了dataset_manifest.csv文件，里面包含了所有样本的标签和位置信息。
# # 要根据这些信息生成用于目标检测的数据集，包括训练集、验证集和测试集，以及class-2的弱监督数据集。
############## flchen 2026.10 #######################
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths and detector settings
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"
INFERENCE2_PATH = DATA_DIR / "inference_2.csv"

# Input width of cloud data: 132 x 164
IMAGE_WIDTH = 164

# Legacy/original OD metadata: 164 px -> 41 cells -> about 4 px per cell
# Current 82-bin detector recomputes targets from continuous position_x;
# these legacy cell fields are retained only for backward compatibility.
NUM_CELLS = 41
CELL_WIDTH = IMAGE_WIDTH / NUM_CELLS


# ============================================================
# 1. Output files
# ============================================================
OUTPUT_TRAIN = DATA_DIR / "detection_train.csv"
OUTPUT_VAL = DATA_DIR / "detection_val.csv"
OUTPUT_TEST = DATA_DIR / "detection_test.csv"

# class-2 has no manual excitation_position GT.
# Keep it separate for weak/semi-supervised localization or challenge analysis.
OUTPUT_CLASS2_TRAIN_WEAK = (
    DATA_DIR / "detection_class2_train_weak.csv"
)
OUTPUT_CLASS2_VAL_WEAK = (
    DATA_DIR / "detection_class2_val_weak.csv"
)
OUTPUT_CLASS2_TEST_CHALLENGE = (
    DATA_DIR / "detection_class2_test_challenge.csv"
)

# Fixed held-out class-9 pool selected before model evaluation.
OUTPUT_CLASS9_INFERENCE2_WEAK = (
    DATA_DIR / "detection_class9_inference2_weak.csv"
)

DETECTION_COLUMNS = [
    "file_name",
    "relative_path",
    "source_class",
    "label_v3",
    "split",

    "objectness",
    "position_valid",
    "position_x",
    "position_x_norm",

    "cell_index",
    "cell_offset",

    "excitation_quality",
    "excitation_quality_value",
    "excitation_PIE",
    "excitation_PIE_value",
]

WEAK_COLUMNS = [
    "file_name",
    "relative_path",
    "source_class",
    "label_v3",
    "split",
    "excitation_quality",
    "excitation_quality_value",
    "excitation_PIE",
    "excitation_PIE_value",
]


# ============================================================
# 2. Helpers
# ============================================================
def require_columns(df, columns):
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "dataset_manifest.csv is missing columns: "
            f"{missing}\n"
            "Run prepare_dataset.py first."
        )


def add_detection_targets(df):
    """
    Strict supervised localization targets.

    class-0:
        objectness = 0
        position_valid = 0
        position_x = NaN

    class-1:
        objectness = 1
        position_valid = 1
        position_x = excitation_position_x

    class-2:
        NOT included here because it has no manual position GT.
    """
    out = df.copy()

    out["objectness"] = (
        out["label_v3"] == 1
    ).astype(int)

    out["position_valid"] = (
        (out["label_v3"] == 1)
        & out["excitation_position_x"].notna()
    ).astype(int)

    out["position_x"] = np.where(
        out["position_valid"] == 1,
        out["excitation_position_x"].astype(float),
        np.nan,
    )

    out["position_x_norm"] = (
        out["position_x"] / (IMAGE_WIDTH - 1)
    )

    out["cell_index"] = np.nan
    out["cell_offset"] = np.nan

    positive = out["position_valid"] == 1

    x = out.loc[
        positive,
        "position_x",
    ].to_numpy(dtype=float)

    cell_index = np.floor(
        x / CELL_WIDTH
    ).astype(int)

    cell_index = np.clip(
        cell_index,
        0,
        NUM_CELLS - 1,
    )

    cell_left = cell_index * CELL_WIDTH

    cell_offset = (
        (x - cell_left) / CELL_WIDTH
    )

    cell_offset = np.clip(
        cell_offset,
        0.0,
        1.0,
    )

    out.loc[
        positive,
        "cell_index",
    ] = cell_index

    out.loc[
        positive,
        "cell_offset",
    ] = cell_offset

    return out


def validate_detection_split(df, split_name):
    """
    Strict detection split must contain only class-0 and class-1.

    Every class-1 sample must have excitation_position_x.
    """
    labels = set(
        df["label_v3"]
        .unique()
        .tolist()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{split_name} contains labels other than 0/1: "
            f"{sorted(labels)}"
        )

    class1 = df[
        df["label_v3"] == 1
    ]

    missing_position = int(
        class1["excitation_position_x"]
        .isna()
        .sum()
    )

    if missing_position:
        raise ValueError(
            f"{split_name} has {missing_position} class-1 samples "
            "without excitation_position_x."
        )


def save_csv(df, path, columns):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df[columns].sort_values(
        "file_name"
    ).to_csv(
        path,
        index=False,
    )


def print_stats(name, df):
    counts = (
        df["label_v3"]
        .value_counts()
        .sort_index()
        .to_dict()
        if len(df)
        else {}
    )

    print(
        f"{name:<28} "
        f"total={len(df):>5}  "
        f"labels={counts}"
    )


# ============================================================
# 3. Main
# ============================================================
def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MANIFEST_PATH}\n"
            "Run prepare_dataset.py first."
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "file_name",
        "relative_path",
        "source_class",
        "label_v3",
        "split",
        "excitation_position_x",
        "excitation_quality",
        "excitation_quality_value",
        "excitation_PIE",
        "excitation_PIE_value",
    ]

    require_columns(
        df,
        required_columns,
    )

    df["label_v3"] = pd.to_numeric(
        df["label_v3"],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # 1) Strict supervised detector:
    #    class-0 + class-1 only
    # --------------------------------------------------------
    strict = df[
        df["label_v3"].isin([0, 1])
        & df["split"].isin(
            ["train", "val", "test"]
        )
    ].copy()

    train_df = strict[
        strict["split"] == "train"
    ].copy()

    val_df = strict[
        strict["split"] == "val"
    ].copy()

    test_df = strict[
        strict["split"] == "test"
    ].copy()

    validate_detection_split(
        train_df,
        "detection_train",
    )
    validate_detection_split(
        val_df,
        "detection_val",
    )
    validate_detection_split(
        test_df,
        "detection_test",
    )

    train_df = add_detection_targets(
        train_df
    )
    val_df = add_detection_targets(
        val_df
    )
    test_df = add_detection_targets(
        test_df
    )

    # --------------------------------------------------------
    # 2) class-2 weak/challenge pools
    # --------------------------------------------------------
    class2_train_weak = df[
        (df["label_v3"] == 2)
        & (df["split"] == "train")
    ].copy()

    class2_val_weak = df[
        (df["label_v3"] == 2)
        & (df["split"] == "val")
    ].copy()

    class2_test_challenge = df[
        (df["label_v3"] == 2)
        & (df["split"] == "test")
    ].copy()

    # --------------------------------------------------------
    # 3) class-9 inference_2 held-out pool
    # --------------------------------------------------------
    # inference_2.csv is used only as a fixed roster of file_name values.
    # Canonical metadata are recovered from dataset_manifest.csv.
    if not INFERENCE2_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INFERENCE2_PATH}\n"
            "Create ./data/inference_2.csv first."
        )

    inference2_roster = pd.read_csv(INFERENCE2_PATH)

    if "file_name" not in inference2_roster.columns:
        raise KeyError(
            "inference_2.csv must contain a 'file_name' column."
        )

    inference2_names = (
        inference2_roster["file_name"]
        .astype(str)
        .str.strip()
    )

    if inference2_names.eq("").any():
        raise ValueError(
            "inference_2.csv contains empty file_name values."
        )

    if inference2_names.duplicated().any():
        duplicated = (
            inference2_names[
                inference2_names.duplicated(keep=False)
            ]
            .tolist()[:10]
        )
        raise ValueError(
            "inference_2.csv contains duplicated file_name values: "
            f"{duplicated}"
        )

    selected_names = set(inference2_names.tolist())
    manifest_names = set(df["file_name"].astype(str).tolist())

    unknown = sorted(selected_names - manifest_names)

    if unknown:
        raise ValueError(
            "Some inference_2 files are not present in dataset_manifest.csv. "
            f"Count={len(unknown)}, examples={unknown[:10]}"
        )

    class9_inference2_weak = df[
        df["file_name"].astype(str).isin(selected_names)
    ].copy()

    if len(class9_inference2_weak) != len(selected_names):
        raise RuntimeError(
            "Unexpected inference_2 roster mismatch after manifest lookup."
        )

    invalid_label = class9_inference2_weak[
        class9_inference2_weak["label_v3"] != 9
    ]

    if len(invalid_label):
        raise ValueError(
            "inference_2.csv must contain class-9 samples only."
        )

    invalid_split = class9_inference2_weak[
        class9_inference2_weak["split"] != "inference"
    ]

    if len(invalid_split):
        raise ValueError(
            "inference_2 samples must come from the original 'inference' split."
        )

    # Explicitly mark this manually fixed held-out subset.
    class9_inference2_weak["split"] = "inference2"

    # --------------------------------------------------------
    # 3) Leakage check for strict detector
    # --------------------------------------------------------
    strict_sets = {
        "train": set(train_df["file_name"]),
        "val": set(val_df["file_name"]),
        "test": set(test_df["file_name"]),
    }

    names = list(strict_sets)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = (
                strict_sets[a]
                & strict_sets[b]
            )

            if overlap:
                raise RuntimeError(
                    f"Detection leakage between {a}/{b}: "
                    f"{len(overlap)} overlapping files."
                )

    # --------------------------------------------------------
    # 4) Save
    # --------------------------------------------------------
    save_csv(
        train_df,
        OUTPUT_TRAIN,
        DETECTION_COLUMNS,
    )

    save_csv(
        val_df,
        OUTPUT_VAL,
        DETECTION_COLUMNS,
    )

    save_csv(
        test_df,
        OUTPUT_TEST,
        DETECTION_COLUMNS,
    )

    save_csv(
        class2_train_weak,
        OUTPUT_CLASS2_TRAIN_WEAK,
        WEAK_COLUMNS,
    )

    save_csv(
        class2_val_weak,
        OUTPUT_CLASS2_VAL_WEAK,
        WEAK_COLUMNS,
    )

    save_csv(
        class2_test_challenge,
        OUTPUT_CLASS2_TEST_CHALLENGE,
        WEAK_COLUMNS,
    )

    save_csv(
        class9_inference2_weak,
        OUTPUT_CLASS9_INFERENCE2_WEAK,
        WEAK_COLUMNS,
    )

    # --------------------------------------------------------
    # 5) Summary
    # --------------------------------------------------------
    print("=" * 78)
    print("Detection dataset preparation")
    print("=" * 78)

    print_stats(
        "detection_train",
        train_df,
    )
    print_stats(
        "detection_val",
        val_df,
    )
    print_stats(
        "detection_test",
        test_df,
    )

    print_stats(
        "class2_train_weak",
        class2_train_weak,
    )
    print_stats(
        "class2_val_weak",
        class2_val_weak,
    )
    print_stats(
        "class2_test_challenge",
        class2_test_challenge,
    )

    print_stats(
        "class9_inference2_weak",
        class9_inference2_weak,
    )

    print()
    print("Saved:")
    for path in [
        OUTPUT_TRAIN,
        OUTPUT_VAL,
        OUTPUT_TEST,
        OUTPUT_CLASS2_TRAIN_WEAK,
        OUTPUT_CLASS2_VAL_WEAK,
        OUTPUT_CLASS2_TEST_CHALLENGE,
        OUTPUT_CLASS9_INFERENCE2_WEAK,
    ]:
        print(f"  {path}")

    print()
    print("Training semantics:")
    print("  class-0 -> objectness=0, no position loss")
    print("  class-1 -> objectness=1, position loss enabled")
    print("  class-2 -> no strict position GT; excluded from strict detector loss")
    print("  class-2 weak/challenge files remain separate from strict detector training")
    print("  class-9 inference2 -> fixed held-out pool for blind manual annotation")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
