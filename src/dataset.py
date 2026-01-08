"""
Dataset and dataloader for RSNA mammography experiments.

Key features:
- CancerDataset: loads PNG images and integer cancer labels.
- TransformingSubset: wraps a subset of indices with its own transform.
- get_dataloaders_rsna: patient + laterality grouped split, with optional small balanced warm-up subset and optional weighted sampling.

"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as transforms
import pandas as pd
import torch
import numpy as np 
import random
import os
import json 



class CancerDataset(Dataset):
    def __init__(self, image_dir, label_dict, transform=None):
        self.image_dir = Path(image_dir) 
        self.image_files = sorted(self.image_dir.glob('**/*.png')) # collect and sort all the PNG files
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB') # open the files in RGB
        label = self.label_dict[img_path.stem] # use filename's stem to match the label
        if self.transform:
            img = self.transform(img) # apply augmentation
        label = torch.tensor(label, dtype=torch.long)  
        return img, label
    
from torch.utils.data import Dataset

class TransformingSubset(Dataset):
    """Picklable subset that applies its own transform on top of a base dataset."""
    def __init__(self, indices, transform, base):
        self.idx = list(indices)
        self.tf = transform
        self.base = base

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img, y = self.base[self.idx[i]]  # base returns (PIL-transformed or raw) tensor,label
        return self.tf(img), y

def _calc_prevalence(indices, base, fn2label):
    labels = []
    for idx in indices:
        stem = base.image_files[idx].stem
        labels.append(fn2label.get(stem, 0))
    labels = np.array(labels, dtype=int)
    if len(labels) == 0:
        return 0.0
    return labels.mean()

def get_dataloaders_rsna(
    image_dir,
    label_csv_path,
    batch_size=128,
    val_pct=0.2,
    num_workers=4,
    use_weighted_sampler=False,
    seed=42,
    return_small_loader=False,
):
    """
    Build train/val DataLoaders for the RSNA dataset with:
    - patient_id + laterality grouped split (to avoid leakage between breasts),
    - fixed split saved/loaded from JSON for reproducibility,
    - optional 1:3 (pos:neg) small warm-up subset,
    - optional WeightedRandomSampler to handle class imbalance.

    Args:
        image_dir: directory with preprocessed PNGs (filename stem = image_id).
        label_csv_path: path to train.csv including columns:
                        patient_id, image_id, laterality, cancer.
        val_pct: approximate validation fraction at group level.
        use_weighted_sampler: if True, uses WeightedRandomSampler for train.
        return_small_loader: if True, also returns a small balanced train loader.

    Returns:
        If return_small_loader:
            small_train_loader, train_loader, val_loader
        else:
            train_loader, val_loader
    """
    df=pd.read_csv(label_csv_path) # load label csv and normalize datatypes
    df=df[["patient_id","image_id","laterality","cancer"]].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["laterality"] = df["laterality"].astype(str)
    df["cancer"] = df["cancer"].astype(int)
    
    fn2label = {img_id: label for img_id, label in zip(df["image_id"], df["cancer"])} # map img_id -> cancer label
    base = CancerDataset(image_dir, fn2label, transform=None)
    stem2idx = {p.stem: i for i, p in enumerate(base.image_files)} # map filename stem -> index in base.image_files
    
    group2indices = {}
    group_labels = {}
    missing = 0
    grouped = df.groupby(["patient_id", "laterality"], sort=False) # grouped by patient id and laterality to avoid images of same breast stay in both train and val

    for (pid, lat), g in grouped:
        idxs = []
        # Align the dataset csv with the actual images in disk
        for img_id in g["image_id"].values:
            stem = str(img_id)
            if stem in stem2idx:
                idxs.append(stem2idx[stem])
            else: # image filename listed in csv but not found as PNG
                missing += 1
        if idxs:
            key = f"{pid}_{lat}"
            group2indices[key] = idxs
            group_labels[key]=int(g["cancer"].max()) # if any image in the group is positive, the group is positive (any case diagnosed as cancer, the breast is diagnosed as cancer)

    if missing > 0:
        print(f"[get_dataloaders_rsna] Warning: {missing} images in csv not found under {image_dir}")

    total_samples = sum(len(v) for v in group2indices.values()) # for logging
    # Define the name of split JSON and where to store
    split_dir = os.path.dirname(label_csv_path)
    split_name = f"rsna_split_seed{seed}.json"
    split_path = os.path.join(split_dir, split_name) 
    # Split group keys into positive and negative
    pos_groups = [gk for gk, lab in group_labels.items() if lab == 1]
    neg_groups = [gk for gk, lab in group_labels.items() if lab == 0]


    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        train_group_keys = set(split["train_group_keys"])
        val_group_keys   = set(split["val_group_keys"])

        # Sanity check
        # Make sure all the group keys still in the current data
        all_keys = set(group2indices.keys())
        missing_in_current = (train_group_keys | val_group_keys) - all_keys
        if missing_in_current:
            print(f"[RSNA split] WARNING: some groups in split file not found in current data: "
                f"{len(missing_in_current)} groups")
        # Check whether there exist new group in current data
        missing_in_split = all_keys - (train_group_keys | val_group_keys)
        if missing_in_split:
            print(f"[RSNA split] WARNING: current data has {len(missing_in_split)} new groups "
                f"not in split file; they will be assigned to train.")
            train_group_keys = train_group_keys | missing_in_split

        print(f"[RSNA split] Loaded fixed split from {split_path}")
        
        n_val_pos = sum(group_labels[gk] == 1 for gk in val_group_keys)
        n_val_neg = sum(group_labels[gk] == 0 for gk in val_group_keys)

    else:

    # Construct training set and validation set and keep the original prevalence
        rng = random.Random(seed)
        rng.shuffle(pos_groups)
        rng.shuffle(neg_groups)
        
        # Desired numbers of positive/negative groups in validation
        n_val_pos = max(1, int(round(len(pos_groups) * val_pct))) if len(pos_groups) > 0 else 0
        n_val_neg = max(1, int(round(len(neg_groups) * val_pct))) if len(neg_groups) > 0 else 0
        # Avoid moving all positive or all negative groups into validation
        n_val_pos = min(n_val_pos, max(len(pos_groups) - 1, 0)) if len(pos_groups) > 1 else n_val_pos
        n_val_neg = min(n_val_neg, max(len(neg_groups) - 1, 0)) if len(neg_groups) > 1 else n_val_neg
        # Build val and derive train
        val_group_keys = set(pos_groups[:n_val_pos] + neg_groups[:n_val_neg])
        train_group_keys = set(group2indices.keys()) - val_group_keys
        
        # Save the split for future runs
        split = {
                "train_group_keys": sorted(list(train_group_keys)),
                "val_group_keys": sorted(list(val_group_keys)),
        }
        with open(split_path, "w") as f:
            json.dump(split, f)
        print(f"[RSNA split] Created new split and saved to {split_path}")
        print(f"[RSNA split] pos_total={len(pos_groups)}, neg_total={len(neg_groups)}, "
                f"val_pos_groups={n_val_pos}, val_neg_groups={n_val_neg}")
        
    val_idx_set = set()
    train_idx_set = set()
    for gk in train_group_keys:
        train_idx_set.update(group2indices[gk])
    for gk in val_group_keys:
        val_idx_set.update(group2indices[gk]) #collect sample index for train and val

    # Safety: Avoid some idx are not included in any group
    all_assigned = val_idx_set | train_idx_set
    all_indices = set(stem2idx.values())
    rest = all_indices - all_assigned
    train_idx_set.update(rest)

    train_indices = sorted(train_idx_set)
    val_indices   = sorted(val_idx_set)

    print(f"[get_rsna_dataloaders_grouped] total_samples={total_samples}, "
        f"train={len(train_indices)}, val={len(val_indices)}, "
        f"val_pct~={len(val_indices) / max(total_samples,1):.3f}")
    print(f"  groups: pos_total={len(pos_groups)}, neg_total={len(neg_groups)}, "
        f"val_pos_groups={n_val_pos}, val_neg_groups={n_val_neg}")
    
    # Optional: Build and save 1:3 warm-up subset (for staged training) 
    # Small subset: all positive groups in train + 3 times of number of positive group negative groups in train
    split_dir = os.path.dirname(label_csv_path)
    small_ratio = 3.0 # 1:3 pos/neg by group
    small_split_name = f"rsna_small_subset_pos1_neg{int(small_ratio)}_seed{seed}.json"
    small_split_path = os.path.join(split_dir, small_split_name)

    if os.path.exists(small_split_path):
        print(f"[RSNA small subset] Found existing 1:{int(small_ratio)} subset: {small_split_path}")
    else:

        # Divide train in pos/neg groups
        train_pos_groups = [gk for gk in train_group_keys if group_labels.get(gk, 0) == 1]
        train_neg_groups = [gk for gk in train_group_keys if group_labels.get(gk, 0) == 0]

        n_pos = len(train_pos_groups)
        n_neg = len(train_neg_groups)

        if n_pos == 0:
            print("[RSNA small subset] WARNING: no positive groups in train; not creating subset.") #sanity check
        else:
            target_neg_groups = int(round(n_pos * small_ratio))
            target_neg_groups = max(1, min(target_neg_groups, n_neg)) # define the desired number of negative sample

            # Fix the random seed to make sure select the same negative groups
            rng_small = random.Random(seed + 999)
            neg_shuffled = train_neg_groups.copy()
            rng_small.shuffle(neg_shuffled)
            chosen_neg_groups = neg_shuffled[:target_neg_groups]

            small_group_keys = sorted(list(train_pos_groups + chosen_neg_groups))

            # Collect indices of all the samples in the subset
            small_idx_set = set()
            for gk in small_group_keys:
                idxs = group2indices[gk]
                small_idx_set.update(idxs)

            small_train_indices = sorted(list(small_idx_set))
            small_prev = _calc_prevalence(small_train_indices, base, fn2label)

            small_info = {
                "seed": seed,
                "val_pct": val_pct,
                "pos_neg_ratio": [1, small_ratio],
                "train_pos_groups": n_pos,
                "train_neg_groups": n_neg,
                "small_group_keys": small_group_keys,      
                "small_num_samples": len(small_train_indices),
                "small_prevalence": float(small_prev),
            }

            with open(small_split_path, "w") as f:
                json.dump(small_info, f, indent=2)

            print(f"[RSNA small subset] Saved 1:{int(small_ratio)} subset to {small_split_path}")
            print(f"[RSNA small subset] groups: train_pos={n_pos}, train_neg={n_neg}, "
                f"small_groups={len(small_group_keys)}")
            print(f"[RSNA small subset] samples: {len(small_train_indices)}, "
                f"prevalence={small_prev:.4f}")

    train_prev = _calc_prevalence(train_indices, base, fn2label)
    val_prev   = _calc_prevalence(val_indices, base, fn2label)
    print(f"  prevalence: train={train_prev:.4f}, val={val_prev:.4f}") # log the prevalence


    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ]) # resize, normalize and apply augmentation for training data

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ]) # only resize and normalize validation data

    train_ds = TransformingSubset(train_indices, train_tf, base)
    val_ds   = TransformingSubset(val_indices,   val_tf,   base)

    # Optional: Use WeightedRandomSampler to oversample minority class in train
    if use_weighted_sampler:
        train_labels = []
        for idx in train_ds.idx:  
            img_path = base.image_files[idx]
            stem = img_path.stem
            train_labels.append(fn2label[stem])

        train_labels = np.array(train_labels, dtype=int)

        class_sample_count = np.bincount(train_labels, minlength=2)
        class_sample_count[class_sample_count == 0] = 1

        weights_per_class = 1.0 / class_sample_count

        sample_weights = weights_per_class[train_labels]
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    small_train_loader = None
    if return_small_loader: # rebuild the small subset
        small_ratio = 3.0
        split_dir = os.path.dirname(label_csv_path)
        small_split_name = f"rsna_small_subset_pos1_neg{int(small_ratio)}_seed{seed}.json"
        small_split_path = os.path.join(split_dir, small_split_name)

        if os.path.exists(small_split_path):
            with open(small_split_path, "r") as f:
                small_info = json.load(f)

            small_group_keys = small_info.get("small_group_keys", [])
            small_idx_set = set()
            for gk in small_group_keys:
                idxs = group2indices.get(gk)
                if idxs is not None:
                    small_idx_set.update(idxs)

            small_train_indices = sorted(list(small_idx_set))
            print(f"[RSNA small subset] loading DataLoader: {len(small_train_indices)} samples")

            small_train_ds = TransformingSubset(small_train_indices, train_tf, base)

            small_train_loader = DataLoader(
                small_train_ds,
                batch_size=batch_size,
                shuffle=True,       
                num_workers=num_workers,
                pin_memory=True,
            )

        else:
            print(f"[RSNA small subset] WARNING: JSON not found at {small_split_path}")
            small_train_loader = None

        return small_train_loader, train_loader, val_loader
    
    return train_loader, val_loader

