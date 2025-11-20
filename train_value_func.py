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

# --- 0. New Sampler Definition (Implementation of Pi-Star/QT-Opt Strategy) ---
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Ensure batch_size is even for 50/50 split
        if self.batch_size % 2 != 0:
            print(f"Warning: Batch size {self.batch_size} is odd. Rounding down for balanced split.")
        
        # Get labels from dataset (pre-calculated in dataset __init__)
        targets = np.array(self.dataset.sample_labels)
        
        self.neg_indices = np.where(targets == 0)[0]  # Failure (Bin 0)
        self.pos_indices = np.where(targets > 0)[0]   # Success (Bin > 0)
        
        # Strategy: Use the larger set as the baseline length to ensure we see all data
        # The smaller set will cycle (repeat) within the epoch
        self.n_samples = max(len(self.neg_indices), len(self.pos_indices))
        self.half_batch = self.batch_size // 2
        self.n_batches = self.n_samples // self.half_batch
        
        print(f"Balanced Sampler Initialized:")
        print(f"  - Failure samples: {len(self.neg_indices)}")
        print(f"  - Success samples: {len(self.pos_indices)}")
        print(f"  - Total Batches per Epoch: {self.n_batches}")

    def __iter__(self):
        # Shuffle indices at the start of each epoch
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
            
            # Note: batch_sampler yields a LIST of indices
            yield batch_indices

    def __len__(self):
        return self.n_batches

# --- 1. Model Definition: Multi-View Fusion ---
class MultiViewValueModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=201):
        super().__init__()
        print(f"Creating Multi-View Model with backbone: {backbone_name}")
        
        # Shared visual backbone (remove classification head)
        self.backbone = create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        
        # Fusion head: 3 image features -> num_classes
        input_dim = self.feat_dim * 3
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes) # Output Logits
        )

    def forward(self, img_room, img_left_wrist, img_right_wrist):
        # Extract features (shared weights)
        f_room = self.backbone(img_room)
        f_left = self.backbone(img_left_wrist)
        f_right  = self.backbone(img_right_wrist)
        
        # Concatenate
        concat_feat = torch.cat([f_room, f_left, f_right], dim=1)
        
        # Predict
        logits = self.head(concat_feat)
        return logits

# --- 2. Dataset Definition ---
class RecapDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, fold=0, is_training=True, failure_multiplier=1.5, num_classes=201):
        self.root_dir = root_dir
        self.transform = transform
        self.min_val = -failure_multiplier
        self.max_val = 0.0
        self.num_classes = num_classes
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Split by Fold
        if is_training:
            self.data = df[df['fold'] != fold].reset_index(drop=True)
        else:
            self.data = df[df['fold'] == fold].reset_index(drop=True)
            
        # Pre-calculate labels for weighted sampling (0=Failure, 1=Success)
        self.sample_labels = []
        for val in self.data['normalized_value']:
            if abs(val - self.min_val) < 1e-6:
                self.sample_labels.append(0) # Failure
            else:
                self.sample_labels.append(1) # Success
            
        print(f"Dataset loaded: {len(self.data)} samples (Training: {is_training})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        ep_id = f"episode_{int(row['episode_id']):04d}"
        fr_id = f"{int(row['frame_id']):06d}"
        status = row['status'] # 'success' or 'failure'
        
        path_base = os.path.join(self.root_dir, status, ep_id)
        
        # Cam 0: room, Cam 1: left wrist, Cam 2: right wrist
        p_room = os.path.join(path_base, f"{fr_id}_color_0.jpg") 
        p_left = os.path.join(path_base, f"{fr_id}_color_1.jpg")
        p_right  = os.path.join(path_base, f"{fr_id}_color_2.jpg")

        # Load images
        img_room = Image.open(p_room).convert('RGB')
        img_left = Image.open(p_left).convert('RGB')
        img_right = Image.open(p_right).convert('RGB')

        if self.transform:
            img_room = self.transform(img_room)
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        # Process Label
        raw_val = float(row['normalized_value'])
        val = max(self.min_val, min(self.max_val, raw_val))
        norm = (val - self.min_val) / (self.max_val - self.min_val)
        label = int(round(norm * (self.num_classes - 1)))
        label = max(0, min(self.num_classes - 1, label))
        
        return img_room, img_left, img_right, label

# --- 3. Main Training Logic ---
_logger = logging.getLogger('train')

def main():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0, help='Validation fold index')
    parser.add_argument('--multiplier', type=float, default=1.5, help='Failure penalty multiplier')
    parser.add_argument('--num-classes', type=int, default=201, help='Number of classes/bins')
    
    # Model
    parser.add_argument('--model', default='efficientnet_b0', type=str)
    parser.add_argument('--batch-size', type=int, default=8) # Keeping 8 is fine for high-res
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    
    # System
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', default='./output', type=str)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    utils.setup_default_logging()
    log_path = os.path.join(args.output, 'train.log')
    
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    _logger.info(f"Logging to {log_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CONSTANTS ---
    IMG_HEIGHT = 480
    IMG_WIDTH = 640
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # --- TRAINING TRANSFORMS (Robot-Safe) ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(IMG_HEIGHT, IMG_WIDTH), 
            scale=(0.90, 1.0), 
            ratio=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05), 
            scale=None, 
            shear=None
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # --- VALIDATION TRANSFORMS ---
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 2. Datasets
    dataset_train = RecapDataset(
        args.csv, args.data_dir, transform=train_transform, 
        fold=args.fold, is_training=True, failure_multiplier=args.multiplier,
        num_classes=args.num_classes
    )
    dataset_val = RecapDataset(
        args.csv, args.data_dir, transform=val_transform, 
        fold=args.fold, is_training=False, failure_multiplier=args.multiplier,
        num_classes=args.num_classes
    )

    # --- MODIFIED: Balanced Batch Sampler ---
    # Replaced WeightedRandomSampler with BalancedBatchSampler
    train_batch_sampler = BalancedBatchSampler(dataset_train, batch_size=args.batch_size)

    # Use batch_sampler in DataLoader
    # Note: batch_size, shuffle, sampler, drop_last are mutually exclusive with batch_sampler
    loader_train = DataLoader(
        dataset_train, 
        batch_sampler=train_batch_sampler, # <--- Using the new sampler
        num_workers=args.workers, 
        pin_memory=True
    )
    
    # Validation loader remains standard
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 3. Model
    model = MultiViewValueModel(backbone_name=args.model, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = create_optimizer_v2(model, opt='adamw', lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.epochs, warmup_epochs=5)
    
    # 5. Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    
    # 6. Training Loop
    _logger.info(f"Start training for {args.epochs} epochs...")
    
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_m = utils.AverageMeter()
        
        train_bar = tqdm(loader_train, desc=f"Train Epoch {epoch}", leave=False, ncols=100)
        for batch_idx, (img_f, img_w, img_s, target) in enumerate(train_bar):
            img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward
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
        acc_m = utils.AverageMeter()
        mae_m = utils.AverageMeter()
        
        fail_acc_m = utils.AverageMeter()
        succ_mae_m = utils.AverageMeter()
        fail_loss_m = utils.AverageMeter()
        succ_loss_m = utils.AverageMeter()
        
        bin_values = torch.linspace(dataset_train.min_val, dataset_train.max_val, args.num_classes).to(device).unsqueeze(0)
        
        with torch.no_grad():
            for (img_f, img_w, img_s, target) in loader_val:
                img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
                target = target.to(device)
                
                output = model(img_f, img_w, img_s)
                loss_per_sample = criterion(output, target)
                loss = loss_per_sample.mean()
                
                acc1, _ = utils.accuracy(output, target, topk=(1, 5))
                
                probs = torch.softmax(output, dim=1)
                pred_val = torch.sum(probs * bin_values, dim=1)
                target_val = bin_values[0, target]
                mae = torch.abs(pred_val - target_val).mean()
                
                # Metrics split
                fail_mask = (target == 0)
                succ_mask = ~fail_mask
                pred_cls = output.argmax(dim=1)
                
                if fail_mask.sum() > 0:
                    # Failure is correct if predicted bin is 0 (or close to 0, e.g., <=2)
                    fail_correct = (pred_cls[fail_mask] == 0).float().mean()
                    fail_acc_m.update(fail_correct.item(), fail_mask.sum().item())
                    fail_loss_m.update(loss_per_sample[fail_mask].mean().item(), fail_mask.sum().item())
                
                if succ_mask.sum() > 0:
                    succ_mae = torch.abs(pred_val[succ_mask] - target_val[succ_mask]).mean()
                    succ_mae_m.update(succ_mae.item(), succ_mask.sum().item())
                    succ_loss_m.update(loss_per_sample[succ_mask].mean().item(), succ_mask.sum().item())

                val_loss_m.update(loss.item(), img_f.size(0))
                acc_m.update(acc1.item(), img_f.size(0))
                mae_m.update(mae.item(), img_f.size(0))
        
        current_mae = mae_m.avg
        _logger.info(
            f"Epoch {epoch} Eval: "
            f"Loss {val_loss_m.avg:.4f}, "
            f"MAE {current_mae:.4f}, "
            f"Fail-Acc {fail_acc_m.avg*100:.1f}%, "
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
