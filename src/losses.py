import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torchvision.ops import sigmoid_focal_loss

def compute_prevalence_and_pos_weight(
    train_csv_path: str,
    device: torch.device,
    column: str = "cancer",
    eps: float = 1e-6, # avoid the situation that prevalence is too low
):
    """
    Read prevalence from train.csv and calculate pos_weight
    Return: (prevalence, pos_weight_tensor)
    """
    df = pd.read_csv(train_csv_path)
    df = df[[column]].copy()
    df[column] = df[column].astype(int)
    prev = df[column].mean() 

    prev_clamped = max(prev, eps)
    ratio = (1.0 - prev_clamped) / prev_clamped  # neg/pos ratio

    pos_weight_value = math.sqrt(ratio)
    pos_weight = torch.tensor([pos_weight_value], device=device)

    print(f"[pos_weight] prevalence={prev:.6f}, ratio={(ratio):.2f}, "
        f"using pos_weight={pos_weight_value:.3f}")

    return prev, pos_weight

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # float or tensor or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits (N,) or (N,1)，targets 0/1
        if logits.dim() > 1:
            logits = logits.view(-1)
        targets = targets.view(-1).float()

        alpha = self.alpha
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=logits.device)

        loss = sigmoid_focal_loss(
            logits,
            targets,
            alpha=alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss

def build_criterion(
    loss_type: str,
    device: torch.device,
    train_csv_path: str | None = None,
    gamma: float = 2.0,
    focal_alpha: float | None = None,
):
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type in ["bce_pos", "focal", "focal_pos"]:
        if train_csv_path is None:
            raise ValueError(f"loss_type={loss_type} needs train_csv_path to evaluate the data imbalance")

        prevalence, pos_weight = compute_prevalence_and_pos_weight(train_csv_path, device)

        if loss_type == "bce_pos":
            print(f"[loss: bce_pos] prevalence={prevalence:.6f}, pos_weight={pos_weight.item():.3f}")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # focal loss
        if focal_alpha is None:
            # fewer pos sample，bigger alpha
            alpha = 1.0 - prevalence
        else:
            alpha = focal_alpha

        if loss_type == "focal_pos":
            alpha = float(pos_weight.item() / (1.0 + pos_weight.item()))

        print(f"[loss: {loss_type}] prevalence={prevalence:.6f}, alpha={alpha:.3f}, gamma={gamma}")

        return FocalLossWithLogits(alpha=alpha, gamma=gamma)

    raise ValueError(f"Unknown loss_type: {loss_type}")