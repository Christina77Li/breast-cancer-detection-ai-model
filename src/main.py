import wandb
from datetime import datetime
import os
import torch
import math 
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd 
import numpy as np

from model import get_model, load_model, unfreeze_layers
from dataset import get_dataloaders_rsna
from train import train_baseline
from losses import build_criterion

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")


os.chdir(os.path.dirname(os.path.abspath(__file__)))
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_images_dir = os.path.join(INPUT_DIR, "images_as_pngs_512", "train_images_processed_512")
    train_csv_path   = os.path.join(INPUT_DIR, "train.csv")
    small_train_dl, train_dl, val_dl = get_dataloaders_rsna(train_images_dir, train_csv_path, batch_size=128, return_small_loader=True)

    model = get_model(num_classes=1).to(device)

    # Using BCEWithLogitsLoss with pos_weight
    criterion = build_criterion(loss_type="bce_pos", device=device, train_csv_path=train_csv_path,)

    #Using Focal Loss
    #criterion = build_criterion(
    #    loss_type="focal_pos",    # or "focal"
    #    device=device,
    #    train_csv_path=train_csv_path,
    #    gamma=2.0,
    #)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=5e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,        # gentler than 0.1
        patience=3,
        threshold=1e-4,    # ignore tiny jitter
        cooldown=1,        # brief cooldown after a reduction
        min_lr=1e-8,
        verbose=True
    )

    run = wandb.init(
        project="breast-cancer-detection",         
        name=f"RSNA_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "resnet101",
            "num_classes": 1,
            "epochs": 100,
            "batch_size": getattr(train_dl, "batch_size", None),
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "scheduler": "ReduceLROnPlateau(patience=3,factor=0.5)",
            "freeze": "unfrozen",
            "train_dir": "./input/train_images_processed_512",
            "train_csv": "./input/train.csv",
        },
        resume="allow",   # enables resuming if run was interrupted
    )

    wandb.watch(model, log="all", log_freq=100)

    train_baseline(model, train_dl, val_dl, num_epochs=50, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, wandb_run=run)

    if os.path.exists("best_model.pth"):
        art = wandb.Artifact("best_resnet101", type="model")
        art.add_file("best_model.pth")
        wandb.log_artifact(art)

    wandb.finish()

if __name__ == "__main__":
    # Windows-safe entry; prevents re-running script inside worker processes
    import torch.multiprocessing as mp
    mp.freeze_support()
    # Optional: keep W&B from trying a separate service process
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    # If your script assumes CWD = repo root:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main()