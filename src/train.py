"""
Training for the breast cancer classification experiments.

This module implements:

- train_one_epoch:
    A single training epoch over one dataloader. Works with logits-based
    binary losses such as BCEWithLogitsLoss or focal loss.

- evaluate_metrics:
    Evaluation on a dataloader with many metrics:
    loss, accuracy@0.5, recall@0.5, F1@0.5, ROC-AUC, PR-AUC,
    best F1 over all thresholds, sensitivity/specificity at 0.5, prevalence.

- train_baseline:
    Baseline training loop that:
      * trains all parameters jointly,
      * uses a scheduler (e.g. ReduceLROnPlateau) on validation AUC,
      * applies early stopping on AUC,
      * and saves multiple "best" checkpoints:
        best loss / acc / F1 / AUC / PR-AUC,
        plus top-k checkpoints by AUC.

- train_unfreeze:
    Staged freeze–unfreeze training ("SAFE-UNFREEZE"):
      * warm-up by training only the final fully connected head, keep backbone BatchNorm layers frozen,
      * then gradually unfreeze ResNet blocks (layer4 → layer3 → layer2 → layer1 → conv1)
        when validation AUC reaches a plateau,
      * assign separate learning rates to each newly unfrozen block,
      * still use early stopping and checkpointing as in the baseline.

Helper components:

- TopKKeeper:
    Maintains the top-k checkpoints according to a metric (e.g. AUC),
    automatically deleting worse checkpoints from disk.

- freeze_backbone_bn:
    Utility to freeze BatchNorm2d layers in the backbone (eval mode + no grad).

- make_optimizer_per_layer / count_trainable_params:
    Convenience functions for per-layer optimizers and param counting.

All training loops work with:
    - a model that outputs one logit per image (shape [B] or [B, 1]),
    - dataloaders yielding (image, label) for binary classification.
Optional Weights & Biases logging is supported via `wandb_run`.
"""
import torch
import torch.nn as nn
import wandb
import os
import math
import heapq
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

eps = 1e-6
SNAPSHOT_EVERY = 5

def _save(model, path):
    torch.save(model.state_dict(), path)
    print(f"  >> saved: {path}")

class TopKKeeper:
    """Keep top-k checkpoints by a metric (higher is better)."""
    def __init__(self, k: int, prefix: str):
        self.k = k
        self.prefix = prefix
        self.heap = []  # (metric, path)
        self.counter=0

    def try_add(self, model, metric_value: float, epoch: int):
        if not (metric_value is not None and math.isfinite(metric_value)):
            return
        fname = f"{self.prefix}_e{epoch:03d}_{metric_value:.5f}_{self.counter:02d}.pth"
        self.counter+=1
        if len(self.heap) < self.k:
            _save(model, fname)
            heapq.heappush(self.heap, (metric_value, fname))
        else:
            if metric_value > self.heap[0][0] + 1e-8:
                _, worst_path = heapq.heappop(self.heap)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                _save(model, fname)
                heapq.heappush(self.heap, (metric_value, fname))

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs) # logits, shape [B] or [B, 1]
        outputs = outputs.squeeze(-1) # ensure shape [B]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Accumulate total loss (sum over samples, to later compute mean)
        running_loss += loss.item() * inputs.size(0)
        # Convert logits -> probabilities -> binary predictions with th=0.5
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()
        # Update accuracy stats
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_metrics(model, dataloader, criterion, device, sweep_steps=1001):
    model.eval()

    total_loss, total_n = 0.0, 0
    all_logits, all_targets = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()           # BCEWithLogitsLoss needs float {0,1}

        logits = model(inputs)                       # [B] 或 [B,1]
        logits = logits.squeeze(-1)                  

        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n    += bs

        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits = torch.cat(all_logits)                   # [N]
    y      = torch.cat(all_targets)                  # [N], float {0,1}
    probs  = torch.sigmoid(logits)                   # probabilities [0,1]
    y_np   = y.numpy()
    p_np   = probs.numpy()

    # Fixed threshold 0.5
    pred05 = (probs >= 0.5).int().numpy()
    acc    = accuracy_score(y_np, pred05)
    rec    = recall_score(y_np, pred05, zero_division=0)
    f1_05  = f1_score(y_np, pred05, zero_division=0)
    
    # ROC-AUC and PR-AUC
    try:
        auc    = roc_auc_score(y_np, p_np)
    except ValueError:
        auc = float('nan')
    try:
        pr_auc = average_precision_score(y_np, p_np)
    except ValueError:
        pr_auc = float('nan')

    # Sweep thresholds in [0,1] to find the best F1
    ths = torch.linspace(0, 1, steps=sweep_steps)
    best_f1, best_th = 0.0, 0.5
    for th in ths:
        pred = (probs >= th).int().numpy()
        f1   = f1_score(y_np, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)

    # Confusion matrix at threshold 0.5 for sensitivity / specificity
    cm = confusion_matrix(y_np, pred05, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens_den=tp+fn
    spec_den=tn+fp
    sens = (tp / sens_den) if sens_den>0 else 0.0
    spec = (tn / spec_den) if spec_den>0 else 0.0

    return {
        "loss": total_loss / max(total_n, 1),
        "acc@0.5": acc,
        "recall@0.5": rec,
        "f1@0.5": f1_05,
        "auc": auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_th": best_th,
        "sens@0.5": sens,
        "spec@0.5": spec,
        "n": int(total_n),
        "prevalence": float(y.mean().item()),
    }

def train_baseline(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, wandb_run=None):

    best = {"loss": float("inf"), "acc": -1.0, "f1": -1.0, "auc": -1.0, "pr_auc": -1.0}
    topk_auc = TopKKeeper(k=3, prefix="ckpt_auc")
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, start_epoch=5)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vm = evaluate_metrics(model, val_loader, criterion, device)  # single evaluation call on val
        val_loss = float(vm["loss"])
        val_acc  = float(vm["acc@0.5"])
        val_f1   = float(vm["f1@0.5"])
        val_rec  = float(vm["recall@0.5"])        
        val_auc  = float(vm.get("auc", float('nan')))
        val_pr   = float(vm.get("pr_auc", float('nan')))
        best_th  = float(vm.get("best_th", 0.5))
        best_f1  = float(vm.get("best_f1", 0.0))  
        sens     = float(vm.get("sens@0.5", 0.0))
        spec     = float(vm.get("spec@0.5", 0.0))
        prev     = float(vm.get("prevalence", 0.0))

        if wandb_run is not None:
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy@0.5": val_acc,
                "val/f1@0.5": val_f1,
                "val/recall@0.5": val_rec,       
                "val/auc": val_auc,
                "val/pr_auc": val_pr,
                "val/best_th": best_th,
                "val/best_f1": best_f1,          
                "lr": optimizer.param_groups[0]["lr"],
            })

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc@0.5: {val_acc:.4f} "
            f"F1@0.5: {val_f1:.4f} Recall@0.5: {val_rec:.4f} AUC: {val_auc:.4f} PR-AUC: {val_pr:.4f}"
        )
        print(
            f"  LR: {optimizer.param_groups[0]['lr']:.8f} | "
            f"val_prev: {prev:.4f} sens@0.5: {sens:.4f} spec@0.5: {spec:.4f} | "
            f"best_F1: {best_f1:.4f} @ th={best_th:.3f}"
        )

        # Scheduler steps on AUC 
        if scheduler is not None:
            metric_for_scheduler = val_auc
            if math.isnan(metric_for_scheduler):
                # If AUC is undefined this epoch, step with -inf so scheduler will reduce LR in 'max' mode
                metric_for_scheduler = float("-inf")
            scheduler.step(metric_for_scheduler)
        
        # Update best-checkpoint logic for various metrics
        if val_loss + eps < best["loss"]:
            best["loss"] = val_loss
            _save(model, f"best_loss_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_loss", "val/loss": val_loss})

        if val_acc > best["acc"] + eps:
            best["acc"] = val_acc
            _save(model, f"best_acc_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_acc", "val/accuracy": val_acc})

        if val_f1 > best["f1"] + eps:
            best["f1"] = val_f1
            _save(model, f"best_f1_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_f1", "val/f1": val_f1})

        if not math.isnan(val_auc) and val_auc > best["auc"] + eps:
            best["auc"] = val_auc
            _save(model, f"best_auc_e{epoch+1:03d}.pth")
            import json
            with open(f"best_auc_e{epoch+1:03d}.pth.metrics.json", "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "auc": val_auc,
                    "pr_auc": val_pr,
                    "best_th": best_th
                }, f, indent=2)
            if wandb_run: wandb.log({"event": "save_best_auc", "val/auc": val_auc, "val/best_th": best_th})
        # Top-k AUC checkpoints (kept on disk by TopKKeeper)
        if not math.isnan(val_pr) and val_pr > best["pr_auc"] + eps:
            best["pr_auc"] = val_pr
            _save(model, f"best_pr_auc_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_pr_auc", "val/pr_auc": val_pr})

        if not math.isnan(val_auc):
            topk_auc.try_add(model, val_auc, epoch+1)

        # Periodic snapshots every SNAPSHOT_EVERY epochs
        if (epoch + 1) % SNAPSHOT_EVERY == 0:
            _save(model, f"snapshot_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "snapshot"})
        
        # Early stopping based on AUC
        early_stopping(val_auc, epoch)
        if early_stopping.save_best:
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Best model saved.")
            if wandb_run:
                wandb.log({"event": "best_model_saved", "val_auc": val_auc})
                
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Optional: Adjust the per-layer optimizers automatically (we set lr manually)
def make_optimizer_per_layer(model, lr_map, weight_decay=1e-4): 
    param_groups = []
    for name, lr in lr_map.items():
        mod = getattr(model, name, None)
        if mod is None: 
            continue
        wd_params, no_wd_params = [], []
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:            
                no_wd_params.append(p)
            else:
                wd_params.append(p)
        if wd_params:
            param_groups.append({"params": wd_params, "lr": lr, "weight_decay": weight_decay})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "lr": lr, "weight_decay": 0.0})
    return torch.optim.Adam(param_groups)

def freeze_backbone_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()                 
        for p in module.parameters():
            p.requires_grad = False   

def train_unfreeze(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, wandb_run=None):
    # Warmup：only train the final fully-connected head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    # Use a relatively large LR for the head at the beginning
    # Start with CosineAnnealingLR so we don't get aggressively cut by Plateau too early
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    early_stopping = EarlyStopping(patience=10, min_delta=0.002, start_epoch=25, mode='max')
    best = {"loss": float("inf"), "acc": 0.0, "f1": 0.0, "auc": 0.0, "pr_auc": 0.0} #track best values
    topk_auc = TopKKeeper(k=3, prefix="ckpt_auc") #keep tok-k checkpoints by AUC
    SNAPSHOT_EVERY = 5
    eps = 1e-6

    # UNFREEZE control hyperparameters
    WARMUP_EPOCHS = 15           # train only the head for at least this many epochs
    MIN_AUC_GAIN = 0.0025          # if improvement < this, treat as plateau
    COOLDOWN_AFTER_UNFREEZE = 12  # epochs to wait after each unfreeze
    all_blocks = ['layer4', 'layer3', 'layer2','layer1', 'conv1']  

    # Per-block learning rates for newly unfrozen layers (earlier layers get smaller LR)
    LAYER_LR_MAP = {
        'layer4': 5e-5,
        'layer3': 1e-5,
        'layer2': 5e-6,
        'layer1': 2e-7,
        'conv1' : 1e-7,
    }

    # Head learning rate after each unfreeze (downscaled to reduce gradient conflicts)
    HEAD_LR_MAP = {
        'layer4': 5e-5,
        'layer3': 3e-5,
        'layer2': 2e-5,
        'layer1': 1e-5,
        'conv1' : 1e-5,
    }

    # State variables for staged unfreezing
    unfreeze_index = 0 # which block in all_blocks will be unfrozen next
    cooldown = 0 # cooldown counter after each unfreeze
    best_auc_seen = -1.0 # best validation AUC observed so far

    for epoch in range(num_epochs):
        # -----1. Training Phase-----
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # -----2. Validation Phase-----
        vm = evaluate_metrics(model, val_loader, criterion, device)  # 只调用一次
        val_loss = float(vm["loss"])
        val_acc  = float(vm["acc@0.5"])
        val_f1   = float(vm["f1@0.5"])
        val_auc  = float(vm.get("auc", float('nan')))
        val_pr   = float(vm.get("pr_auc", float('nan')))
        best_th  = float(vm.get("best_th", 0.5))
        
        # -----3. UNFREEZE Phase-----
        # Update "plateau vs improvement" status for AUC
        if val_auc > best_auc_seen:
            best_auc_seen = val_auc
            auc_improved = True
        else:
            # Allow small fluctuations; do not treat as regression unless the drop exceeds MIN_AUC_GAIN
            auc_improved = (best_auc_seen - val_auc) < MIN_AUC_GAIN 

        # Trigger safe unfreezing (layer4 -> layer3 -> layer2 -> layer1 -> conv1)
        if ((epoch + 1) >= WARMUP_EPOCHS) and (not auc_improved) and (cooldown == 0) and (unfreeze_index < len(all_blocks)):
            block = all_blocks[unfreeze_index]

            # 1) Unfreeze parameters of this block
            target_module = getattr(model, block)
            for p in target_module.parameters():
                p.requires_grad = True

            # 2) Freeze backbone BatchNorm stats to avoid drifting running stats
            model.apply(freeze_backbone_bn)

            # 3) Add a new optimizer param group for this block with its own LR
            lr_new = LAYER_LR_MAP.get(block, 1e-5)
            new_params = [p for p in target_module.parameters() if p.requires_grad]
            optimizer.add_param_group({"params": new_params, "lr": lr_new, "weight_decay": 1e-4})
            if isinstance(scheduler, ReduceLROnPlateau): # re-create to match the new number of param groups
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=6, factor=0.3, verbose=True)

            # 4) Lower the head LR according to the mapping for this block
            head_lr = HEAD_LR_MAP.get(block, optimizer.param_groups[0]["lr"])
            optimizer.param_groups[0]["lr"] = head_lr
            cooldown = COOLDOWN_AFTER_UNFREEZE
            unfreeze_index += 1
            
            print(f"  >> SAFE-UNFREEZE: unfroze {block}, lr_new={lr_new:.1e}, head_lr={head_lr:.1e}")
            if wandb_run:
                wandb.log({
                    "event": "unfreeze",
                    "block": block,
                    "unfreeze_step": unfreeze_index,
                    "trainable_params": count_trainable_params(model)
                })

        # -----4. Scheduler step-----
        # CosineAnnealingLR: step() every epoch；ReduceLROnPlateau: step() on validation AUC
        if isinstance(scheduler, ReduceLROnPlateau):
            try:
                scheduler.step(val_auc)
            except IndexError: # safety net in case a new param group was added this epoch
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=6, factor=0.3, verbose=True) # recreate ReduceLROnPlateau
                scheduler.step(val_auc)
        else:
            scheduler.step()

        # -----5. Logging to Weights & Biases-----
        if wandb_run is not None:
            wandb.log({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/f1@0.5": val_f1,
                "val/auc": val_auc,
                "val/pr_auc": val_pr,
                "val/best_th": best_th,
                "lr/group_0": optimizer.param_groups[0]["lr"],
                })

            # Also log LR for all param groups (head + newly unfrozen blocks)
            for i, g in enumerate(optimizer.param_groups):
                wandb.log({f"lr/group_{i}": g["lr"]})

        # Console printout
        lr_info = ", ".join([f"g{i}:{g['lr']:.1e}" for i, g in enumerate(optimizer.param_groups)])
        print(f"[Epoch {epoch+1}] Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1@0.5 {val_f1:.4f} AUC {val_auc:.4f} PR-AUC {val_pr:.4f}")
        print(f"  LRs: {lr_info}")

        # -----6. Save various best checkpoints-----
        if val_loss + eps < best["loss"]:
            best["loss"] = val_loss
            _save(model, f"best_loss_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_loss", "val/loss": val_loss})

        if val_acc > best["acc"] + eps:
            best["acc"] = val_acc
            _save(model, f"best_acc_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_acc", "val/accuracy": val_acc})

        if val_f1 > best["f1"] + eps:
            best["f1"] = val_f1
            _save(model, f"best_f1_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_f1", "val/f1": val_f1})

        if not math.isnan(val_auc) and val_auc > best["auc"] + eps:
            best["auc"] = val_auc
            _save(model, f"best_auc_e{epoch+1:03d}.pth")
            import json
            # Sync metrics (including best_th) into a JSON file）
            with open(f"best_auc_e{epoch+1:03d}.pth.metrics.json", "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "auc": val_auc,
                    "pr_auc": val_pr,
                    "best_th": best_th
                }, f, indent=2)
            if wandb_run: wandb.log({"event": "save_best_auc", "val/auc": val_auc, "val/best_th": best_th})
        
        # Maintain top-k checkpoints by AUC
        if not math.isnan(val_pr) and val_pr > best["pr_auc"] + eps:
            best["pr_auc"] = val_pr
            _save(model, f"best_pr_auc_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "save_best_pr_auc", "val/pr_auc": val_pr})

        if not math.isnan(val_auc):
            topk_auc.try_add(model, val_auc, epoch+1)

        # -----7. Periodic snapshots-----
        if (epoch + 1) % SNAPSHOT_EVERY == 0:
            _save(model, f"snapshot_e{epoch+1:03d}.pth")
            if wandb_run: wandb.log({"event": "snapshot"})

        # -----8. Cooldown countdown and scheduler switch-----
        if cooldown > 0:
            cooldown -= 1
        # After all blocks have been unfrozen and cooldown finished,if we are still using CosineAnnealingLR, switch to ReduceLROnPlateau
        if (unfreeze_index == len(all_blocks)) and (cooldown == 0) and isinstance(scheduler, CosineAnnealingLR):
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=6, factor=0.3, verbose=True)
            print("  >> Scheduler switched to ReduceLROnPlateau(mode='max', patience=6)")

        # -----9. Early stopping on validation AUC-----
        early_stopping(val_auc, epoch)
        if early_stopping.save_best:
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Best model saved.")
            if wandb_run:
                wandb.log({"event": "best_model_saved", "val_auc": val_auc})
