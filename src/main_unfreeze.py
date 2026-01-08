import wandb
from datetime import datetime
import os
import torch

from model import get_model, load_model
from dataset import get_dataloaders_rsna
from train import train_unfreeze
from losses import build_criterion

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_images_dir = os.path.join(INPUT_DIR, "images_as_pngs_512", "train_images_processed_512")
    train_csv_path   = os.path.join(INPUT_DIR, "train.csv")
    small_train_dl, train_dl, val_dl = get_dataloaders_rsna(train_images_dir, train_csv_path, batch_size=128, return_small_loader=True)
    
    model = load_model("./best_auc_model.pth", device, num_classes=1)
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=3e-4, weight_decay=1e-4)

    # Using BCEWithLogitsLoss with pos_weight
    criterion = build_criterion(loss_type="bce_pos", device=device, train_csv_path=train_csv_path,)

    #Using Focal Loss
    #criterion = build_criterion(
    #    loss_type="focal_pos",    # or focal
    #    device=device,
    #    train_csv_path=train_csv_path,
    #    gamma=2.0,
    #)

    run = wandb.init(
        project="breast-cancer-detection",         
        name=f"resnet101_staged_unfreeze_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={"model": "resnet101",
                "optimizer": "adam",
                "epochs": 50,
                "loss": "BCEWithLogitsLoss",
                "train_images": "./input/train_images_processed_512",
                "train_csv": "./input/train.csv"
        },
        resume=False, 
    )

    wandb.watch(model, log="all", log_freq=100)

    train_unfreeze(model, train_dl, val_dl, num_epochs=50, criterion=criterion, optimizer=optimizer, device=device, wandb_run=run)

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
