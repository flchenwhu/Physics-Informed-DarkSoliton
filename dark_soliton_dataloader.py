#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dark_soliton_dataset.py
=======================

Dark Soliton 项目的统一 PyTorch Dataset / DataLoader 数据接口。
"""

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SUPPORTED_TASKS = {"classification", "detection", "unlabeled"}
SUPPORTED_NORMALIZATIONS = {
    "none", "sample_zscore", "sample_minmax", "dataset_zscore"
}


def _safe_float(value, default=np.nan):
    if value is None:
        return float(default)
    try:
        if pd.isna(value):
            return float(default)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=-1):
    if value is None:
        return int(default)
    try:
        if pd.isna(value):
            return int(default)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


class DarkSolitonDataset(Dataset):
    """Dark Soliton 统一 Dataset。"""

    def __init__(
        self,
        csv_path: Union[str, Path],
        task: str = "classification",
        project_root: Optional[Union[str, Path]] = None,
        image_key: str = "cloud data",
        normalize: str = "none",
        dataset_mean: Optional[float] = None,
        dataset_std: Optional[float] = None,
        transform: Optional[Callable] = None,
        check_file_exists: bool = True,
        expected_shape=(132, 164),
    ):
        super().__init__()

        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 不存在：{self.csv_path}")

        task = str(task).lower().strip()
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"不支持 task={task!r}，支持：{sorted(SUPPORTED_TASKS)}"
            )

        normalize = str(normalize).lower().strip()
        if normalize not in SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"不支持 normalize={normalize!r}，支持：{sorted(SUPPORTED_NORMALIZATIONS)}"
            )

        self.task = task
        self.image_key = image_key
        self.normalize = normalize
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.transform = transform
        self.check_file_exists = bool(check_file_exists)
        self.expected_shape = tuple(expected_shape) if expected_shape else None

        if project_root is None:
            # 默认 CSV 位于 <project_root>/data/*.csv
            self.project_root = self.csv_path.parent.parent
        else:
            self.project_root = Path(project_root).expanduser().resolve()

        if self.normalize == "dataset_zscore":
            if dataset_mean is None or dataset_std is None:
                raise ValueError(
                    "normalize='dataset_zscore' 时必须提供 dataset_mean 和 dataset_std。"
                )
            if float(dataset_std) <= 0:
                raise ValueError("dataset_std 必须 > 0。")

        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise RuntimeError(f"CSV 没有数据：{self.csv_path}")

        self._validate_columns()
        self.paths = self._build_paths()

        if self.check_file_exists:
            self._validate_paths()

    def _validate_columns(self):
        required = {"file_name", "relative_path"}

        if self.task == "classification":
            if "training_label" not in self.df.columns and "label_v3" not in self.df.columns:
                required.add("label_v3")

        elif self.task == "detection":
            required.update({
                "label_v3", "objectness", "position_valid",
                "position_x", "position_x_norm", "cell_index", "cell_offset",
            })

        missing = required - set(self.df.columns)
        if missing:
            raise KeyError(
                f"{self.csv_path.name} 缺少字段：{sorted(missing)}"
            )

    def _build_paths(self):
        paths = []
        for _, row in self.df.iterrows():
            relative = str(row.get("relative_path", "")).strip()
            if relative and relative.lower() != "nan":
                path = Path(relative)
                if not path.is_absolute():
                    path = self.project_root / path
                paths.append(path)
                continue

            source_class = str(row.get("source_class", "")).strip()
            file_name = str(row["file_name"]).strip()
            paths.append(
                self.project_root / "data_files" / source_class / file_name
            )
        return paths

    def _validate_paths(self):
        missing = [str(path) for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.csv_path.name} 中有 {len(missing)} 个文件不存在。\n"
                f"示例：{missing[:5]}"
            )

    def _load_image(self, path: Path) -> np.ndarray:
        obj = np.load(path, allow_pickle=True)

        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            obj = obj.item()

        if isinstance(obj, dict):
            candidates = [
                self.image_key,
                self.image_key.replace(" ", "_"),
                self.image_key.replace("_", " "),
            ]
            image = None
            for key in candidates:
                if key in obj:
                    image = np.asarray(obj[key], dtype=np.float32)
                    break
            if image is None:
                raise KeyError(
                    f"{path.name}: 找不到 {self.image_key!r}；可用 keys={list(obj.keys())}"
                )
        elif isinstance(obj, np.ndarray) and obj.ndim == 2:
            image = np.asarray(obj, dtype=np.float32)
        else:
            raise TypeError(f"{path.name}: 无法识别 .npy 结构。")

        if image.ndim != 2:
            raise ValueError(f"{path.name}: 图像不是二维，shape={image.shape}")

        if self.expected_shape is not None and tuple(image.shape) != self.expected_shape:
            raise ValueError(
                f"{path.name}: shape={image.shape}，期望={self.expected_shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(f"{path.name}: 图像中存在 NaN/Inf。")

        return image

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)

        if self.normalize == "none":
            return image

        if self.normalize == "sample_zscore":
            mean = float(image.mean())
            std = float(image.std())
            if std < 1e-8:
                return (image - mean).astype(np.float32)
            return ((image - mean) / std).astype(np.float32)

        if self.normalize == "sample_minmax":
            min_value = float(image.min())
            max_value = float(image.max())
            scale = max_value - min_value
            if scale < 1e-8:
                return np.zeros_like(image, dtype=np.float32)
            return ((image - min_value) / scale).astype(np.float32)

        if self.normalize == "dataset_zscore":
            return (
                (image - float(self.dataset_mean)) / float(self.dataset_std)
            ).astype(np.float32)

        raise RuntimeError(f"Unexpected normalization: {self.normalize}")

    def _get_label(self, row):
        if self.task == "unlabeled":
            return -1

        if "training_label" in row.index:
            return _safe_int(row.get("training_label"), default=-1)
        return _safe_int(row.get("label_v3"), default=-1)

    def _optional_physics_targets(self, row):
        position = _safe_float(
            row.get("excitation_position_x", row.get("position_x", np.nan))
        )

        if "position_valid" in row.index:
            position_valid = _safe_int(row.get("position_valid"), default=0)
        else:
            position_valid = int(np.isfinite(position))

        quality = _safe_float(row.get("excitation_quality_value", np.nan))
        quality_valid = int(np.isfinite(quality))

        pie = _safe_int(row.get("excitation_PIE_value", -1), default=-1)
        pie_valid = int(pie >= 0)

        return {
            "position": torch.tensor(
                position if np.isfinite(position) else -1.0,
                dtype=torch.float32,
            ),
            "position_valid": torch.tensor(position_valid, dtype=torch.bool),
            "quality": torch.tensor(
                quality if np.isfinite(quality) else -1.0,
                dtype=torch.float32,
            ),
            "quality_valid": torch.tensor(quality_valid, dtype=torch.bool),
            "pie": torch.tensor(pie, dtype=torch.long),
            "pie_valid": torch.tensor(pie_valid, dtype=torch.bool),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = self.paths[index]

        image = self._load_image(path)
        image = self._normalize_image(image)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        sample = {
            "image": image_tensor,
            "label": torch.tensor(self._get_label(row), dtype=torch.long),
            "file_name": str(row["file_name"]),
            "relative_path": str(row.get("relative_path", "")),
            "source_class": str(row.get("source_class", "")),
            "index": torch.tensor(index, dtype=torch.long),
        }

        if self.task in {"classification", "detection"}:
            sample.update(self._optional_physics_targets(row))

        if self.task == "detection":
            sample.update({
                "objectness": torch.tensor(
                    _safe_float(row.get("objectness"), default=0.0),
                    dtype=torch.float32,
                ),
                "position_norm": torch.tensor(
                    _safe_float(row.get("position_x_norm"), default=-1.0),
                    dtype=torch.float32,
                ),
                "cell_index": torch.tensor(
                    _safe_int(row.get("cell_index"), default=-1),
                    dtype=torch.long,
                ),
                "cell_offset": torch.tensor(
                    _safe_float(row.get("cell_offset"), default=-1.0),
                    dtype=torch.float32,
                ),
            })

        return sample

    def class_counts(self):
        if self.task == "unlabeled":
            return {-1: len(self.df)}

        if "training_label" in self.df.columns:
            series = pd.to_numeric(self.df["training_label"], errors="coerce")
        else:
            series = pd.to_numeric(self.df["label_v3"], errors="coerce")

        return series.value_counts().sort_index().to_dict()


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False,
):
    """建立 DataLoader。Mac 开发阶段建议 num_workers=0。"""
    if num_workers == 2:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


@torch.no_grad()
def compute_dataset_mean_std(
    dataset: DarkSolitonDataset,
    batch_size: int = 64,
    num_workers: int = 2,
):
    """
    计算给定 Dataset 的全局 pixel mean/std。

    推荐：只传入 train.csv 且 normalize="none" 的 Dataset。
    不能使用 val/test/inference 计算训练归一化统计量。
    """
    if dataset.normalize != "none":
        raise ValueError(
            "计算训练集全局 mean/std 时，dataset 必须使用 normalize='none'。"
        )

    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for batch in loader:
        images = batch["image"].double()
        total_sum += images.sum().item()
        total_sq_sum += (images * images).sum().item()
        total_pixels += images.numel()

    if total_pixels == 0:
        raise RuntimeError("Dataset 没有像素数据。")

    mean = total_sum / total_pixels
    variance = total_sq_sum / total_pixels - mean * mean
    variance = max(variance, 0.0)
    std = variance ** 0.5

    return float(mean), float(std)
