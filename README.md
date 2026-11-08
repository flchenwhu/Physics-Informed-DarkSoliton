[README.md](https://github.com/user-attachments/files/34539797/README.md)
# Physics-Informed Multi-Task Learning for Dark Soliton Analysis

This repository provides code, curated annotations, standard-test results, and representative pretrained models for **dark-soliton classification and localization** in experimental Bose–Einstein condensate (BEC) images.

## Overview

We study four learning settings using SimpleCNN, ResNet-18, RepViT-M0.9, and FasterNet-T0:

| Stage | Approach |
|:---:|---|
| 1 | Classification-only baseline |
| 2 | Localization-only baseline |
| 3 | Shared multi-task learning |
| 4 | Physics-informed multi-task learning (proposed) |

The proposed method combines image features with a vertically integrated density profile for joint classification and localization.

## Repository contents

- **Training and evaluation:** `train_*.py`, `evaluate_*.py`, and `dark_soliton_dataloader.py`.
- **Annotations and splits:** `data/`, including curated Gold/Blind-test annotations.
- **Standard Test results:** [`Standard_Test_Results.xlsx`](Standard_Test_Results.xlsx) (readable tables) and [`Standard_Test_Results.csv`](Standard_Test_Results.csv) (machine-readable results).
- **Pretrained models:** `output/seed42/` contains representative best checkpoints for the four backbones and stages. Other seeds can be reproduced with the training code or requested from the authors.
- **Environment:** `requirements_environment_darksoliton.yml` and `requirements_darksoliton.txt`.

## Dataset and usage

The experimental images are from the [NIST Dark Solitons in BECs Dataset](https://doi.org/10.18434/mds2-2363). Raw image files are **not** redistributed here; download them from the original source and arrange them under `data_files/` using the paths referenced by the provided CSV files.

To set up the environment:

```bash
conda env create -f requirements_environment_darksoliton.yml
conda activate darksoliton
```

Run the corresponding stage or evaluation script as needed, for example:

```bash
python train_physics_multitask_v23.py
python evaluate_gold_challenge_stage4_v23.py
```

Select the backbone and random seed in the scripts. When using supplied checkpoints, verify that the checkpoint paths match the `output/seed42/` layout.

## Citation

If you use this repository, please cite the associated paper. Bibliographic details will be added upon publication.
