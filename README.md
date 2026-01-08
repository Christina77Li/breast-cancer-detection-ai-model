# Breast Cancer Detection from Mammography (PyTorch)

This repository contains the training and evaluation code for a deep learning model
for **breast cancer detection from screening mammography**.

Slides from my presentation on this project:
- https://docs.google.com/presentation/d/12bJwUVOoxeTjp7NOjOmRgpeBMi_0jYOmMT9PS8T3LiQ/edit?usp=sharing

This work explores various training strategies to make deep learning models more sample-efficient and to improve their overall classification performance.

---

## Overview

- **Backbone**: ResNet-based classifier (PyTorch, pretrained on ImageNet)  
- **Task**: Binary classification (cancer vs. non-cancer) on 2D mammogram images  
- **Data handling**:
  - images grouped by patient (and laterality) to avoid leakage,
  - fixed split saved to file for reproducibility
- **Imbalance**:
  - weighted loss (e.g., `BCEWithLogitsLoss` with `pos_weight`)
  - optional focal losses
- **Training utilities**:
  - early stopping
  - best-checkpoint saving based on multiple metrics
  - separate script for gradually unfreezing of deeper layers

> **Note:** Image data and CSV files are **not included** in this repository
> due to size. This repo is focused on the code.

---

## Project Structure

```text
src/
  dataset.py          # Dataset & DataLoader builders (patient-level grouping, splits)
  model.py            # Model definition and loading
  train.py            # Core training loop and metric computation
  main.py             # Entry point for baseline training
  main_unfreeze.py    # Entry point for gradually unfreezing training
  evaluate_model.py   # Evaluation script for trained checkpoints
  losses.py           # Loss functions (BCE with pos_weight, focal variants, etc.)
  early_stopping.py   # Early stopping utility
input/                # Local data directory (not tracked by git)
```

## How to Run (code only)

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**

   Place your CSV and processed images under `input/`, for example:

   ```text
   input/
     train.csv
     images_as_pngs_512/
       train_images_processed_512/
         *.png
   ```
   (Paths are configurable in the scripts.)
3. **Start training**
   Baseline training/Subset Warm-up+Full data training:
   ```bash
      cd src
      python main.py
   ```
   Gradually Unfreezing:
   ```bash
      cd src
      python main_unfreeze.py
   ```
4. **Evaluate a checkpoint**
   ```bash
      cd src
      python evaluate_model.py
   ```
---
## Purpose
  This repository is part of my research on *deep learning-based breast cancer diagnosis*, and is published as a portfolio artifact to demonstrate:

  - building a complete PyTorch training pipeline,

  - dealing with highly imbalanced medical imaging data,

  - designing patient-level splits to prevent data leakage,

  - and implementing gradually unfreezing and metric-based early stopping.

