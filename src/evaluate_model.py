import torch
import os
from model import load_model
from train import evaluate_metrics
from dataset import get_dataloaders_rsna
from losses import build_criterion  

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model("./best_model _single01_1e4_1e4.pth", device, num_classes=1)
model.eval()

train_images_dir = "./input/images_as_pngs_512/train_images_processed_512"
train_csv_path   = "./input/train.csv"

# Use the same dataloader & split of training phase
val_dl = get_dataloaders_rsna(
    train_images_dir,
    train_csv_path,
    batch_size=128,
    return_small_loader=False,
)

# Use the same loss function of training phase
criterion = build_criterion(
    loss_type="bce_pos",        # or focal
    device=device,
    train_csv_path=train_csv_path,
)

metrics = evaluate_metrics(model, val_dl, criterion, device)

print(f"Accuracy@0.5: {metrics['acc@0.5']:.4f}")
print(f"Recall@0.5:   {metrics['recall@0.5']:.4f}")
print(f"F1@0.5:       {metrics['f1@0.5']:.4f}")
print(f"AUC:          {metrics['auc']:.4f}")
print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
print(f"Best F1:      {metrics['best_f1']:.4f} @ th={metrics['best_th']:.3f}")
