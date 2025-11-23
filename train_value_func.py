import argparse
import logging
import os
import time
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from timm import utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model

# --- 0. Balanced Sampler (Unchanged) ---
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        if self.batch_size % 2 != 0:
            print(f"Warning: Batch size {self.batch_size} is odd. Rounding down for balanced split.")
        
        # Get labels from dataset (0=Failure Zone, 1=Success Zone)
        targets = np.array(self.dataset.sample_labels)
        
        self.neg_indices = np.where(targets == 0)[0]  # Failure (Value < -1.0)
        self.pos_indices = np.where(targets > 0)[0]   # Success (Value >= -1.0)
        
        self.n_samples = max(len(self.neg_indices), len(self.pos_indices))
        self.half_batch = self.batch_size // 2
        self.n_batches = self.n_samples // self.half_batch
        
        print(f"Balanced Sampler Initialized:")
        print(f"  - Failure samples: {len(self.neg_indices)}")
        print(f"  - Success samples: {len(self.pos_indices)}")
        print(f"  - Total Batches per Epoch: {self.n_batches}")

    def __iter__(self):
        neg_indices_shuffled = self.neg_indices.copy()
        pos_indices_shuffled = self.pos_indices.copy()
        np.random.shuffle(neg_indices_shuffled)
        np.random.shuffle(pos_indices_shuffled)
        
        neg_ptr = 0
        pos_ptr = 0
        
        for _ in range(self.n_batches):
            batch_indices = []
            
            # 1. Pick 50% Failure samples
            for _ in range(self.half_batch):
                idx = neg_indices_shuffled[neg_ptr % len(self.neg_indices)]
                batch_indices.append(idx)
                neg_ptr += 1
                
            # 2. Pick 50% Success samples
            for _ in range(self.batch_size - self.half_batch):
                idx = pos_indices_shuffled[pos_ptr % len(self.pos_indices)]
                batch_indices.append(idx)
                pos_ptr += 1
            
            yield batch_indices

    def __len__(self):
        return self.n_batches

# --- 1. Model Definition (Unchanged) ---
class MultiViewValueModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=201):
        super().__init__()
        print(f"Creating Multi-View Model with backbone: {backbone_name}")
        
        self.backbone = create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        
        input_dim = self.feat_dim * 3
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes) 
        )

    def forward(self, img_room, img_left_wrist, img_right_wrist):
        f_room = self.backbone(img_room)
        f_left = self.backbone(img_left_wrist)
        f_right  = self.backbone(img_right_wrist)
        
        concat_feat = torch.cat([f_room, f_left, f_right], dim=1)
        logits = self.head(concat_feat)
        return logits

# --- 2. Dataset Definition (Updated for RECAP Logic) ---
class RecapDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, fold=0, is_training=True, 
                 min_val=-2.5, max_val=0.0, num_classes=201):
        self.root_dir = root_dir
        self.transform = transform
        
        # RECAP Unified Range
        self.min_val = min_val
        self.max_val = max_val
        self.num_classes = num_classes
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if is_training:
            self.data = df[df['fold'] != fold].reset_index(drop=True)
        else:
            self.data = df[df['fold'] == fold].reset_index(drop=True)
            
        # --- Generate Labels for Balanced Sampler ---
        # Define boundary: Values < -1.0 are considered "Failure/Penalty Zone"
        # Values >= -1.0 are considered "Success Zone"
        self.sample_labels = []
        boundary_threshold = -1.0001 
        
        for val in self.data['normalized_value']:
            if val < boundary_threshold:
                self.sample_labels.append(0) # Failure (Class 0 for sampler)
            else:
                self.sample_labels.append(1) # Success (Class 1 for sampler)
            
        print(f"Dataset loaded: {len(self.data)} samples (Training: {is_training})")
        print(f"Global Value Range: [{self.min_val}, {self.max_val}] -> {self.num_classes} Bins")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        ep_id = f"episode_{int(row['episode_id']):04d}"
        fr_id = f"{int(row['frame_id']):06d}"
        status = row['status'] 
        
        path_base = os.path.join(self.root_dir, status, ep_id)
        
        p_room = os.path.join(path_base, f"{fr_id}_color_0.jpg") 
        p_left = os.path.join(path_base, f"{fr_id}_color_1.jpg")
        p_right  = os.path.join(path_base, f"{fr_id}_color_2.jpg")

        img_room = Image.open(p_room).convert('RGB')
        img_left = Image.open(p_left).convert('RGB')
        img_right = Image.open(p_right).convert('RGB')

        if self.transform:
            img_room = self.transform(img_room)
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        # --- Unified Label Processing ---
        raw_val = float(row['normalized_value'])
        
        # 1. Clip to global range
        val = max(self.min_val, min(self.max_val, raw_val))
        
        # 2. Linear Normalize to 0~1
        norm = (val - self.min_val) / (self.max_val - self.min_val)
        
        # 3. Map to discrete bin
        label = int(round(norm * (self.num_classes - 1)))
        label = max(0, min(self.num_classes - 1, label))
        
        return img_room, img_left, img_right, label

# --- 3. Main Training Logic ---
_logger = logging.getLogger('train')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    
    # RECAP Range Arguments (Replaces multiplier)
    parser.add_argument('--min-val', type=float, default=-2.5, help='Min value (Failure Start), default -2.5')
    parser.add_argument('--max-val', type=float, default=0.0, help='Max value (Success End), default 0.0')
    parser.add_argument('--num-classes', type=int, default=201)
    
    # Model Params
    parser.add_argument('--model', default='efficientnet_b0', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', default='./output', type=str)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    utils.setup_default_logging()
    log_path = os.path.join(args.output, 'train.log')
    
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.info(f"Logging to {log_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CONSTANTS ---
    IMG_HEIGHT = 480
    IMG_WIDTH = 640
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- TRANSFORMS ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(IMG_HEIGHT, IMG_WIDTH), scale=(0.90, 1.0), ratio=(0.95, 1.05), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 2. Datasets
    # Pass min_val and max_val correctly
    dataset_train = RecapDataset(
        args.csv, args.data_dir, transform=train_transform, 
        fold=args.fold, is_training=True, 
        min_val=args.min_val, max_val=args.max_val, num_classes=args.num_classes
    )
    dataset_val = RecapDataset(
        args.csv, args.data_dir, transform=val_transform, 
        fold=args.fold, is_training=False, 
        min_val=args.min_val, max_val=args.max_val, num_classes=args.num_classes
    )

    # 3. Sampler
    train_batch_sampler = BalancedBatchSampler(dataset_train, batch_size=args.batch_size)

    loader_train = DataLoader(
        dataset_train, 
        batch_sampler=train_batch_sampler, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 4. Model
    model = MultiViewValueModel(backbone_name=args.model, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    optimizer = create_optimizer_v2(model, opt='adamw', lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.epochs, warmup_epochs=5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    
    _logger.info(f"Start training for {args.epochs} epochs...")
    best_mae = float('inf')
    
    # Pre-calculate bin values mapping (0..200 -> min..max)
    bin_values = torch.linspace(args.min_val, args.max_val, args.num_classes).to(device).unsqueeze(0)
    
    # Define Failure Boundary (The gap between Success start -1.0 and Failure)
    # Anything predicted below this is considered "Predicting Failure"
    failure_boundary = -1.0 
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_m = utils.AverageMeter()
        
        train_bar = tqdm(loader_train, desc=f"Train Epoch {epoch}", leave=False, ncols=100)
        for batch_idx, (img_f, img_w, img_s, target) in enumerate(train_bar):
            img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(img_f, img_w, img_s) 
            loss_per_sample = criterion(output, target)
            loss = loss_per_sample.mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss_m.update(loss.item(), img_f.size(0))
            train_bar.set_postfix(loss=train_loss_m.avg)
            
        _logger.info(f"Epoch {epoch} Train Loss: {train_loss_m.avg:.4f}")
        scheduler.step(epoch)
        
        # Validation
        model.eval()
        val_loss_m = utils.AverageMeter()
        mae_m = utils.AverageMeter()
        fail_detect_m = utils.AverageMeter() # "Did we correctly see the failure?"
        succ_mae_m = utils.AverageMeter()    # "How close are we on success tracks?"
        
        with torch.no_grad():
            for (img_f, img_w, img_s, target) in loader_val:
                img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
                target = target.to(device)
                
                output = model(img_f, img_w, img_s)
                loss_per_sample = criterion(output, target)
                
                # 1. Predict Value (Expectation)
                probs = torch.softmax(output, dim=1)
                pred_val = torch.sum(probs * bin_values, dim=1) # Float
                
                # 2. Target Value (Float)
                target_val = bin_values[0, target]
                
                # 3. Global MAE
                mae = torch.abs(pred_val - target_val).mean()
                
                # --- RECAP Metrics ---
                # Use the computed float values to separate Success/Failure
                # GT < -1.0 is Failure
                fail_mask = (target_val < failure_boundary)
                succ_mask = ~fail_mask
                
                # Metric A: Failure Detection Rate
                # If GT is Failure, is Pred also Failure (< -1.0)?
                if fail_mask.sum() > 0:
                    detected = (pred_val[fail_mask] < failure_boundary).float().mean()
                    fail_detect_m.update(detected.item(), fail_mask.sum().item())
                
                # Metric B: Success MAE
                # If GT is Success, how close is the value?
                if succ_mask.sum() > 0:
                    succ_mae = torch.abs(pred_val[succ_mask] - target_val[succ_mask]).mean()
                    succ_mae_m.update(succ_mae.item(), succ_mask.sum().item())

                val_loss_m.update(loss_per_sample.mean().item(), img_f.size(0))
                mae_m.update(mae.item(), img_f.size(0))
        
        current_mae = mae_m.avg
        _logger.info(
            f"Epoch {epoch} Eval: "
            f"Loss {val_loss_m.avg:.4f}, "
            f"Global-MAE {current_mae:.4f}, "
            f"Fail-Detect-Rate {fail_detect_m.avg*100:.1f}%, "
            f"Succ-MAE {succ_mae_m.avg:.4f}"
        )
        
        if current_mae < best_mae:
            best_mae = current_mae
            save_path = os.path.join(args.output, f"best_value_net.pth")
            torch.save(model.state_dict(), save_path)
            _logger.info(f"New best model saved with MAE: {best_mae:.4f}")
            
        latest_path = os.path.join(args.output, f"latest_value_net.pth")
        torch.save(model.state_dict(), latest_path)

if __name__ == '__main__':
    main()
