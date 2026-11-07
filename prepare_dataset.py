from pathlib import Path
import ast
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 0. Project paths
# ============================================================
ROOT = Path(__file__).resolve().parent

# Raw npy files:
# ./data_files/class-0/
# ./data_files/class-1/
# ./data_files/class-2/
# ./data_files/class-8/
# ./data_files/class-9/
DATA_ROOT = ROOT / "data_files"

# Metadata / split files / generated CSVs
DATA_DIR = ROOT / "data"

# data_roster.csv: prefer ./data, fallback to project root
ROSTER_CANDIDATES = [
    DATA_DIR / "data_roster.csv",
    ROOT / "data_roster.csv",
]

# Official file lists: support underscore and space variants
TRAIN_LIST_CANDIDATES = [
    DATA_DIR / "train_data(2101-05404).txt",
    DATA_DIR / "train data(2101-05404).txt",
]
TEST_LIST_CANDIDATES = [
    DATA_DIR / "test_data(2101-05404).txt",
    DATA_DIR / "test data(2101-05404).txt",
]


# ============================================================
# 1. Split policy
# ============================================================
SEED = 42

# Supervised classification classes
SUPERVISED_LABELS = {0, 1, 2}

# Special labels
AMBIGUOUS_LABEL = 8
UNLABELED_LABEL = 9

# Official train/test is approximately 90/10.
# Taking 1/9 from official train as validation gives roughly 80/10/10 overall.
VAL_RATIO_FROM_OFFICIAL_TRAIN = 1.0 / 9.0

# Reserve part of class-9 as completely unseen inference data.
INFERENCE_RATIO_FROM_CLASS9 = 0.10

# If you prefer a fixed count, set an integer such as 1000.
# Keep None to use INFERENCE_RATIO_FROM_CLASS9.
INFERENCE_COUNT = None

# Fail immediately if roster files cannot be matched to local npy files.
STRICT_FILE_CHECK = True

# ------------------------------------------------------------
# Project-level expanded class-2 test set
# ------------------------------------------------------------
# We keep the released official train/test txt files unchanged.
#
# After reproducing the original train/validation split, we additionally
# freeze 194 class-2 samples from the PROJECT TRAIN split and move them
# into the PROJECT TEST split.
#
# Expected supervised split after this operation:
#   train: class-0=904, class-1=2570, class-2=633
#   val  : class-0=113, class-1=322,  class-2=103
#   test : class-0=113, class-1=320,  class-2=300
#
# For provenance, the moved 194 samples still keep:
#   official_split = "train"
# while their project split becomes:
#   split = "test"
CLASS2_EXTRA_TEST_N = 194
CLASS2_EXTRA_TEST_SEED = 20260924

# First run: create the frozen roster.
# Later runs: reuse exactly the same 194 filenames.
CLASS2_EXTRA_TEST_ROSTER = (
    DATA_DIR / "class2_extra_test_roster.csv"
)


# ============================================================
# 2. Output paths
# ============================================================
MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

UNLABELED_CSV = DATA_DIR / "unlabeled.csv"
INFERENCE_CSV = DATA_DIR / "inference.csv"

# Compact columns for downstream training
EXPORT_COLUMNS = [
    "file_name",
    "relative_path",
    "source_class",
    "label_v3",
    "training_label",
    "excitation_position",
    "excitation_position_x",
    "excitation_quality",
    "excitation_quality_value",
    "excitation_PIE",
    "excitation_PIE_value",
    "official_split",
    "split",
]


# ============================================================
# 3. Helpers
# ============================================================
def find_existing(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Cannot find {description}. Tried:\n  {tried}"
    )


def read_name_list(path):
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(names) != len(set(names)):
        series = pd.Series(names)
        duplicated = series[series.duplicated()].tolist()
        raise ValueError(
            f"{path.name} contains duplicated filenames, e.g. {duplicated[:5]}"
        )

    return names


def parse_first_number(value):
    """
    Convert values such as:
        [110.35]
        "[110.35]"
        110.35
    to float.

    Return NaN when missing or non-numeric.
    """
    if value is None:
        return np.nan

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return np.nan

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan

    try:
        obj = ast.literal_eval(text)

        if isinstance(obj, (list, tuple, np.ndarray)):
            if len(obj) == 0:
                return np.nan
            return float(obj[0])

        return float(obj)

    except (ValueError, SyntaxError, TypeError):
        return np.nan


def build_npy_index(data_root):
    """
    Scan ./data_files/class-*/*.npy.

    Returns:
        dict[basename] -> absolute Path
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    index = {}
    duplicates = []

    for class_dir in sorted(data_root.glob("class-*")):
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.glob("*.npy")):
            if path.name in index:
                duplicates.append(path.name)
            else:
                index[path.name] = path.resolve()

    if duplicates:
        raise ValueError(
            "Duplicate npy basenames exist across class folders. "
            f"Examples: {duplicates[:5]}"
        )

    if not index:
        raise RuntimeError(f"No .npy files found under {data_root}")

    return index


def add_file_paths(df, npy_index):
    """
    Match data_roster.csv to local files by basename.
    """
    df = df.copy()

    original_names = df["file_name"].astype(str)
    df["roster_file_name"] = original_names
    df["file_name"] = original_names.map(lambda x: Path(x).name)

    if df["file_name"].duplicated().any():
        dup = (
            df.loc[df["file_name"].duplicated(), "file_name"]
            .head()
            .tolist()
        )
        raise ValueError(
            "Basenames in data_roster.csv are not unique. "
            f"Examples: {dup}"
        )

    df["absolute_path"] = df["file_name"].map(
        lambda name: str(npy_index[name]) if name in npy_index else ""
    )
    df["file_exists"] = df["absolute_path"].ne("")

    missing = df.loc[~df["file_exists"], "file_name"].tolist()

    if missing:
        message = (
            f"{len(missing)} roster files were not found under "
            f"./data_files/class-*. Examples: {missing[:5]}"
        )

        if STRICT_FILE_CHECK:
            raise FileNotFoundError(message)

        print(f"[Warning] {message}")

    def to_relative(abs_path):
        if not abs_path:
            return ""

        p = Path(abs_path)

        try:
            return p.relative_to(ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    df["relative_path"] = df["absolute_path"].map(to_relative)

    df["source_class"] = df["absolute_path"].map(
        lambda x: Path(x).parent.name if x else ""
    )

    return df


def validate_official_lists(train_names, test_names, df):
    train_set = set(train_names)
    test_set = set(test_names)

    overlap = train_set & test_set

    if overlap:
        raise ValueError(
            f"Official train/test lists overlap by {len(overlap)} files. "
            f"Examples: {list(overlap)[:5]}"
        )

    roster_names = set(df["file_name"])

    missing_train = train_set - roster_names
    missing_test = test_set - roster_names

    if missing_train:
        raise ValueError(
            f"{len(missing_train)} files in official train list are not "
            f"in data_roster.csv. Examples: {list(missing_train)[:5]}"
        )

    if missing_test:
        raise ValueError(
            f"{len(missing_test)} files in official test list are not "
            f"in data_roster.csv. Examples: {list(missing_test)[:5]}"
        )


def make_validation_split(supervised_official_train):
    """
    Stratified split inside official train only.

    Official test remains untouched.
    """
    train_part, val_part = train_test_split(
        supervised_official_train,
        test_size=VAL_RATIO_FROM_OFFICIAL_TRAIN,
        random_state=SEED,
        stratify=supervised_official_train["label_v3"],
    )

    return train_part.copy(), val_part.copy()


def expand_project_test_with_class2(
    train_df,
    test_df,
):
    """
    Move exactly CLASS2_EXTRA_TEST_N class-2 samples from the already-created
    PROJECT TRAIN split into the PROJECT TEST split.

    Why this function is called AFTER make_validation_split():
        - validation stays identical to the previous protocol;
        - only the project training pool is reduced;
        - original official test samples are all preserved.

    Reproducibility:
        - first run: sample 194 class-2 files using CLASS2_EXTRA_TEST_SEED
          and save them to CLASS2_EXTRA_TEST_ROSTER;
        - later runs: reuse that roster exactly.

    Provenance:
        the moved files keep official_split='train' because that is where the
        released dataset placed them, but their project split becomes 'test'.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    class2_candidates = train_df[
        train_df["label_v3"] == 2
    ].copy()

    if len(class2_candidates) < CLASS2_EXTRA_TEST_N:
        raise ValueError(
            "Not enough class-2 samples in project train split: "
            f"available={len(class2_candidates)}, "
            f"requested={CLASS2_EXTRA_TEST_N}"
        )

    if CLASS2_EXTRA_TEST_ROSTER.exists():
        roster = pd.read_csv(
            CLASS2_EXTRA_TEST_ROSTER
        )

        if "file_name" not in roster.columns:
            raise KeyError(
                f"{CLASS2_EXTRA_TEST_ROSTER.name} must contain "
                "a 'file_name' column."
            )

        selected_names = (
            roster["file_name"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        if len(selected_names) != CLASS2_EXTRA_TEST_N:
            raise ValueError(
                "Frozen class-2 extra-test roster has unexpected size: "
                f"{len(selected_names)} != {CLASS2_EXTRA_TEST_N}"
            )

        if len(selected_names) != len(set(selected_names)):
            raise ValueError(
                "Frozen class-2 extra-test roster contains duplicate filenames."
            )

        candidate_names = set(
            class2_candidates["file_name"]
            .astype(str)
            .tolist()
        )

        missing = sorted(
            set(selected_names) - candidate_names
        )

        if missing:
            raise ValueError(
                "Frozen class-2 extra-test roster contains samples that are "
                "not in the current project-train class-2 pool. "
                f"Examples: {missing[:10]}"
            )

        print(
            "[Info] Reusing frozen class-2 extra-test roster: "
            f"{CLASS2_EXTRA_TEST_ROSTER}"
        )

    else:
        selected = class2_candidates.sample(
            n=CLASS2_EXTRA_TEST_N,
            random_state=CLASS2_EXTRA_TEST_SEED,
            replace=False,
        ).copy()

        selected_names = (
            selected["file_name"]
            .astype(str)
            .sort_values()
            .tolist()
        )

        CLASS2_EXTRA_TEST_ROSTER.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            {
                "file_name": selected_names,
            }
        ).to_csv(
            CLASS2_EXTRA_TEST_ROSTER,
            index=False,
        )

        print(
            "[Info] Created frozen class-2 extra-test roster: "
            f"{CLASS2_EXTRA_TEST_ROSTER}"
        )

    selected_set = set(
        selected_names
    )

    extra_test_df = train_df[
        train_df["file_name"]
        .astype(str)
        .isin(selected_set)
    ].copy()

    remaining_train_df = train_df[
        ~train_df["file_name"]
        .astype(str)
        .isin(selected_set)
    ].copy()

    if len(extra_test_df) != CLASS2_EXTRA_TEST_N:
        raise RuntimeError(
            "Unexpected number of selected extra-test samples: "
            f"{len(extra_test_df)}"
        )

    if not (
        extra_test_df["label_v3"] == 2
    ).all():
        raise RuntimeError(
            "Extra-test subset contains non-class-2 samples."
        )

    expanded_test_df = pd.concat(
        [
            test_df,
            extra_test_df,
        ],
        ignore_index=True,
    )

    if (
        expanded_test_df["file_name"]
        .astype(str)
        .duplicated()
        .any()
    ):
        duplicated = (
            expanded_test_df.loc[
                expanded_test_df["file_name"]
                .astype(str)
                .duplicated(keep=False),
                "file_name",
            ]
            .astype(str)
            .tolist()[:10]
        )

        raise RuntimeError(
            "Expanded project test set contains duplicate files. "
            f"Examples: {duplicated}"
        )

    return (
        remaining_train_df,
        expanded_test_df,
        extra_test_df,
    )


def split_class9(class9_df):
    """
    Split class-9 into:
        unlabeled.csv  -> available to SSL / semi-supervised training
        inference.csv  -> completely held out until final inference
    """
    class9_df = class9_df.copy()

    if len(class9_df) == 0:
        return class9_df.copy(), class9_df.copy()

    rng = np.random.default_rng(SEED)

    if INFERENCE_COUNT is not None:
        n_inference = min(int(INFERENCE_COUNT), len(class9_df))
    else:
        n_inference = int(round(
            len(class9_df) * INFERENCE_RATIO_FROM_CLASS9
        ))

    if len(class9_df) > 1:
        n_inference = max(
            1,
            min(n_inference, len(class9_df) - 1),
        )
    else:
        n_inference = 1

    chosen = rng.choice(
        class9_df.index.to_numpy(),
        size=n_inference,
        replace=False,
    )

    inference_df = class9_df.loc[chosen].copy()
    unlabeled_df = class9_df.drop(index=chosen).copy()

    return unlabeled_df, inference_df


def count_table(df, split_name):
    counts = (
        df["label_v3"]
        .value_counts()
        .sort_index()
        .to_dict()
        if len(df)
        else {}
    )

    return (
        f"{split_name:<12} total={len(df):>5}   "
        f"labels={counts}"
    )


# ============================================================
# 4. Main
# ============================================================
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    roster_path = find_existing(
        ROSTER_CANDIDATES,
        "data_roster.csv",
    )
    train_list_path = find_existing(
        TRAIN_LIST_CANDIDATES,
        "official train_data(2101-05404).txt",
    )
    test_list_path = find_existing(
        TEST_LIST_CANDIDATES,
        "official test_data(2101-05404).txt",
    )

    print("=" * 78)
    print("Dark-soliton dataset preparation")
    print("=" * 78)

    print(f"Roster              : {roster_path}")
    print(f"Official train list : {train_list_path}")
    print(f"Official test list  : {test_list_path}")
    print(f"NPY root            : {DATA_ROOT}")
    print(f"Output directory    : {DATA_DIR}")
    print()

    # --------------------------------------------------------
    # 1) Read roster and official file lists
    # --------------------------------------------------------
    df = pd.read_csv(roster_path)

    required_columns = {
        "file_name",
        "label_v3",
        "excitation_position",
        "excitation_quality",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            "data_roster.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if "excitation_PIE" not in df.columns:
        df["excitation_PIE"] = np.nan

    df["label_v3"] = pd.to_numeric(
        df["label_v3"],
        errors="raise",
    ).astype(int)

    official_train_names = read_name_list(train_list_path)
    official_test_names = read_name_list(test_list_path)

    print(
        f"Official list sizes: "
        f"train={len(official_train_names)}, "
        f"test={len(official_test_names)}"
    )

    # --------------------------------------------------------
    # 2) Match actual npy files
    # --------------------------------------------------------
    npy_index = build_npy_index(DATA_ROOT)

    print(f"Local npy files found: {len(npy_index)}")

    df = add_file_paths(df, npy_index)

    validate_official_lists(
        official_train_names,
        official_test_names,
        df,
    )

    official_train_set = set(official_train_names)
    official_test_set = set(official_test_names)

    df["official_split"] = "unlisted"

    df.loc[
        df["file_name"].isin(official_train_set),
        "official_split",
    ] = "train"

    df.loc[
        df["file_name"].isin(official_test_set),
        "official_split",
    ] = "test"

    # --------------------------------------------------------
    # 3) Parse useful numeric labels
    # --------------------------------------------------------
    df["excitation_position_x"] = (
        df["excitation_position"]
        .map(parse_first_number)
    )

    df["excitation_quality_value"] = (
        df["excitation_quality"]
        .map(parse_first_number)
    )

    df["excitation_PIE_value"] = (
        df["excitation_PIE"]
        .map(parse_first_number)
    )

    df["training_label"] = np.where(
        df["label_v3"].isin(SUPERVISED_LABELS),
        df["label_v3"],
        -1,
    ).astype(int)

    # --------------------------------------------------------
    # 4) Classification train / val / test
    # --------------------------------------------------------
    supervised = df[
        df["label_v3"].isin(SUPERVISED_LABELS)
    ].copy()

    official_train_supervised = supervised[
        supervised["official_split"] == "train"
    ].copy()

    test_df = supervised[
        supervised["official_split"] == "test"
    ].copy()

    supervised_unlisted = supervised[
        supervised["official_split"] == "unlisted"
    ].copy()

    if len(supervised_unlisted):
        print(
            f"[Warning] {len(supervised_unlisted)} supervised "
            "0/1/2 samples are not in official train/test lists. "
            "They remain in manifest but are excluded from "
            "classification train/val/test."
        )

    train_df, val_df = make_validation_split(
        official_train_supervised
    )

    # --------------------------------------------------------
    # 5) Expand PROJECT test with 194 frozen class-2 samples
    # --------------------------------------------------------
    train_df, test_df, class2_extra_test_df = (
        expand_project_test_with_class2(
            train_df,
            test_df,
        )
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Keep an explicit project-test marker in the auxiliary subset too.
    class2_extra_test_df["split"] = "test"

    # --------------------------------------------------------
    # 6) class-9 -> unlabeled + inference
    # --------------------------------------------------------
    class9_df = df[
        df["label_v3"] == UNLABELED_LABEL
    ].copy()

    unlabeled_df, inference_df = split_class9(class9_df)

    unlabeled_df["split"] = "unlabeled"
    inference_df["split"] = "inference"

    # --------------------------------------------------------
    # 7) class-8 kept in manifest only
    # --------------------------------------------------------
    class8_df = df[
        df["label_v3"] == AMBIGUOUS_LABEL
    ].copy()

    class8_df["split"] = "excluded_class8"

    # --------------------------------------------------------
    # 8) Assign final split to full manifest
    # --------------------------------------------------------
    assignment = {}

    for part, split_name in [
        (train_df, "train"),
        (val_df, "val"),
        (test_df, "test"),
        (unlabeled_df, "unlabeled"),
        (inference_df, "inference"),
        (class8_df, "excluded_class8"),
    ]:
        for file_name in part["file_name"]:
            if file_name in assignment:
                raise RuntimeError(
                    f"File assigned to multiple splits: {file_name}"
                )
            assignment[file_name] = split_name

    df["split"] = (
        df["file_name"]
        .map(assignment)
        .fillna("excluded")
    )

    # --------------------------------------------------------
    # 9) Leakage and label checks
    # --------------------------------------------------------
    split_sets = {
        "train": set(train_df["file_name"]),
        "val": set(val_df["file_name"]),
        "test": set(test_df["file_name"]),
        "unlabeled": set(unlabeled_df["file_name"]),
        "inference": set(inference_df["file_name"]),
    }

    split_names = list(split_sets)

    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = split_sets[a] & split_sets[b]

            if overlap:
                raise RuntimeError(
                    f"Data leakage between {a} and {b}: "
                    f"{len(overlap)} overlapping files."
                )

    supervised_class1 = pd.concat(
        [
            train_df[train_df["label_v3"] == 1],
            val_df[val_df["label_v3"] == 1],
            test_df[test_df["label_v3"] == 1],
        ],
        ignore_index=True,
    )

    missing_position = int(
        supervised_class1["excitation_position_x"]
        .isna()
        .sum()
    )

    missing_quality = int(
        supervised_class1["excitation_quality_value"]
        .isna()
        .sum()
    )

    if missing_position:
        print(
            f"[Warning] {missing_position} class-1 samples "
            "are missing excitation_position."
        )

    if missing_quality:
        print(
            f"[Warning] {missing_quality} class-1 samples "
            "are missing excitation_quality."
        )

    # --------------------------------------------------------
    # Extra project-test checks
    # --------------------------------------------------------
    extra_test_names = set(
        class2_extra_test_df["file_name"]
        .astype(str)
        .tolist()
    )

    if extra_test_names & set(
        train_df["file_name"].astype(str)
    ):
        raise RuntimeError(
            "Extra class-2 test files still remain in train_df."
        )

    if extra_test_names & set(
        val_df["file_name"].astype(str)
    ):
        raise RuntimeError(
            "Extra class-2 test files overlap with val_df."
        )

    test_class2_count = int(
        (test_df["label_v3"] == 2).sum()
    )

    if test_class2_count != 300:
        raise RuntimeError(
            "Expected project test to contain exactly 300 class-2 samples, "
            f"got {test_class2_count}."
        )

    # --------------------------------------------------------
    # 10) Save files
    # --------------------------------------------------------
    core_columns = [
        "file_name",
        "relative_path",
        "source_class",
        "label_v3",
        "training_label",
        "official_split",
        "split",
        "excitation_position",
        "excitation_position_x",
        "excitation_quality",
        "excitation_quality_value",
        "excitation_PIE",
        "excitation_PIE_value",
        "roster_file_name",
        "file_exists",
        "absolute_path",
    ]

    remaining_columns = [
        c for c in df.columns
        if c not in core_columns
    ]

    manifest = df[
        core_columns + remaining_columns
    ].copy()

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    train_df[EXPORT_COLUMNS].sort_values(
        "file_name"
    ).to_csv(
        TRAIN_CSV,
        index=False,
    )

    val_df[EXPORT_COLUMNS].sort_values(
        "file_name"
    ).to_csv(
        VAL_CSV,
        index=False,
    )

    test_df[EXPORT_COLUMNS].sort_values(
        "file_name"
    ).to_csv(
        TEST_CSV,
        index=False,
    )

    unlabeled_df[EXPORT_COLUMNS].sort_values(
        "file_name"
    ).to_csv(
        UNLABELED_CSV,
        index=False,
    )

    inference_df[EXPORT_COLUMNS].sort_values(
        "file_name"
    ).to_csv(
        INFERENCE_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # 11) Summary
    # --------------------------------------------------------
    print()
    print("-" * 78)
    print("Split summary")
    print("-" * 78)

    print(count_table(train_df, "train"))
    print(count_table(val_df, "val"))
    print(count_table(test_df, "test"))
    print(count_table(unlabeled_df, "unlabeled"))
    print(count_table(inference_df, "inference"))
    print(count_table(class8_df, "class-8"))

    print()
    print("Saved:")
    for path in [
        MANIFEST_PATH,
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        CLASS2_EXTRA_TEST_ROSTER,
        UNLABELED_CSV,
        INFERENCE_CSV,
    ]:
        print(f"  {path}")

    print()
    print("Notes:")
    print("  1) train/val/test contain only label_v3 = 0/1/2.")
    print("  2) released official train/test txt files are NEVER modified.")
    print(
        "  3) project test = released official test + 194 frozen "
        "class-2 samples moved from the project train split."
    )
    print(
        "  4) moved samples retain official_split='train' for provenance, "
        "but have project split='test'."
    )
    print(
        "  5) expected project test class-2 count = 300 "
        "(106 original + 194 extra)."
    )
    print("  6) unlabeled/inference come from class-9 and are disjoint.")
    print("  7) inference is fully held out from all training stages.")
    print("  8) class-8 stays in dataset_manifest.csv as excluded_class8.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
