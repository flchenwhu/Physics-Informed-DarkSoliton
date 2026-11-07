# Physics-Informed Multi-Task Learning for Dark Soliton Analysis

Code, data annotations, and representative trained models for **classification and localization of dark-soliton excitations** in experimental Bose–Einstein condensate (BEC) images.

## Overview

We compare four learning settings to investigate whether an explicit physical representation improves excitation recognition and localization:

| Stage | Method | Task |
|---|---|---|
| 1 | Classification-only | Classify excitation patterns |
| 2 | Localization-only | Locate excitations |
| 3 | Shared multi-task | Joint classification and localization |
| 4 | Physics-informed multi-task | Joint learning with a density-profile branch |

Stage 4 incorporates a vertically integrated density profile, $P(x)=\sum_y I(y,x)$, into the image-based network. The experiments cover **SimpleCNN, ResNet-18, RepViT-M0.9, and FasterNet-T0**.

## Repository contents

```text
.
├── data/                              # Split CSVs and manual Gold annotations
├── output/seed42/                     # Example trained checkpoints (four backbones)
├── dark_soliton_dataloader.py        # Data loading and preprocessing
├── prepare_dataset.py
├── prepare_detection_dataset.py
├── train_classifier_baseline.py      # Stage 1
├── train_detector_baseline.py        # Stage 2
├── train_multitask_baseline_v23.py   # Stage 3
├── train_physics_multitask_v23.py    # Stage 4
├── evaluate_class9_challenge_stage1.py
├── evaluate_class2_challenge_stage2.py
├── evaluate_gold_challenge_stage3.py
├── evaluate_gold_challenge_stage4_v23.py
├── requirements_darksoliton.txt
└── requirements_environment_darksoliton.yml
```

The `data/` folder includes the **frozen train/validation/test splits**, localization manifests, source metadata, and manually curated Gold annotations. In particular:

- `train.csv`, `val.csv`, `val_v2.csv`, `test.csv`: supervised classification and multi-task splits.
- `detection_train.csv`, `detection_val.csv`, `detection_test.csv`: supervised localization splits.
- `class9_inference2_manual_annotations.csv`: blind classification labels.
- `class2_manual_annotations.csv` and `class2_class9_inference2_manual_annotations.csv`: Gold evaluation annotations.

## Dataset and setup

The experimental images come from the [NIST Dark Solitons in BECs Dataset](https://doi.org/10.18434/mds2-2363). **Raw `.npy` images are not included** in this repository. Download them separately and retain the original filenames:

```text
data_files/
├── class-0/
├── class-1/
├── class-2/
├── class-8/
└── class-9/
```

Create the Python environment using either of the supplied dependency files:

```bash
conda env create -f requirements_environment_darksoliton.yml
conda activate darksoliton
# Alternatively, in a compatible existing environment:
# pip install -r requirements_darksoliton.txt
```

The committed CSVs are intended for reproducing the reported data splits. **Do not rerun the preparation scripts over these files** unless you intend to regenerate them.

## Training and evaluation

Select `MODEL_NAME` and `SEED` near the top of each training script, then run the desired stage:

```bash
python train_classifier_baseline.py
python train_detector_baseline.py
python train_multitask_baseline_v23.py
python train_physics_multitask_v23.py
```

Evaluate the trained models using the corresponding scripts:

```bash
python evaluate_class9_challenge_stage1.py
python evaluate_class2_challenge_stage2.py
python evaluate_gold_challenge_stage3.py
python evaluate_gold_challenge_stage4_v23.py
```

**Evaluation protocol:** classification uses labels for classes 0/1/2; supervised localization uses reliable positions from classes 0/1. Held-out **Gold/Blind test annotations are reserved for final evaluation**, not training or model selection. See the evaluation scripts for the inclusion rules and matching metrics.

## Trained models

To make the repository easier to explore, we provide **seed-42 checkpoints** for all four backbones and four stages. The best checkpoints are stored under:

```text
output/seed42/<backbone>/<stage>/checkpoint/best.pt.tar
```

Here `<backbone>` is `simple_cnn`, `resnet18`, `repvit_m0_9`, or `fasternet_t0`; `<stage>` is `classifier_baseline`, `detector_baseline`, `multitask_baseline`, or `physics_multitask`.

These checkpoints are **representative ready-trained models**, not the complete multi-seed archive. Other random seeds can be reproduced by retraining with the supplied scripts, or the corresponding models may be requested from the authors. If a script expects `output/<backbone>/...`, adjust its checkpoint path to the supplied `output/seed42/<backbone>/...` layout before evaluation.

## Citation

If this code or the curated annotations support your research, please cite the associated paper once its bibliographic information is available. The paper link and citation will be added after publication.
