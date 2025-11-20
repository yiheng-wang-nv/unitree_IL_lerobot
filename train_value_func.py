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
from torch.utils.data import Dataset, DataLoader

from timm import utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model

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
            
        print(f"Dataset loaded: {len(self.data)} samples (Training: {is_training})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Construct image paths (Assuming structure: category/episode_id/frame_id_camera.jpg)
        # Structure: root_dir/status/episode_ID/FRAME_color_CAM.jpg
        # e.g. root_dir/success/episode_0004/000100_color_0.jpg
        
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
        # Mapping float value to class index directly here
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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    
    # System
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='./output', type=str)
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Setup logging
    utils.setup_default_logging()
    log_path = os.path.join(args.output, 'train.log')
    
    # Add file handler to the existing logger
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    _logger.info(f"Logging to {log_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Augmentation (Robot data usually only Normalize and Resize)
    # Mean/std are ImageNet defaults, better recalculate for specific robot scenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets
    dataset_train = RecapDataset(
        args.csv, args.data_dir, transform=transform, 
        fold=args.fold, is_training=True, failure_multiplier=args.multiplier,
        num_classes=args.num_classes
    )
    dataset_val = RecapDataset(
        args.csv, args.data_dir, transform=transform, 
        fold=args.fold, is_training=False, failure_multiplier=args.multiplier,
        num_classes=args.num_classes
    )

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 3. Model
    model = MultiViewValueModel(backbone_name=args.model, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    # 4. Optimizer & Scheduler (using timm utilities)
    optimizer = create_optimizer_v2(model, opt='adamw', lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.epochs, warmup_epochs=5)
    
    # 5. Loss
    # Use label smoothing to handle the ordinal nature of the data better (adjacent bins are related)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 6. Training Loop
    _logger.info(f"Start training for {args.epochs} epochs...")
    
    best_mae = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        # ... (training code remains same) ...
        train_loss_m = utils.AverageMeter()
        
        for batch_idx, (img_f, img_w, img_s, target) in enumerate(loader_train):
            img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(img_f, img_w, img_s) # (B, num_classes)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss_m.update(loss.item(), img_f.size(0))
            
            if batch_idx % 20 == 0:
                _logger.info(f"Epoch {epoch}: [{batch_idx}/{len(loader_train)}] Loss: {train_loss_m.val:.4f} (Avg: {train_loss_m.avg:.4f})")

        scheduler.step(epoch)
        
        # Validation
        model.eval()
        val_loss_m = utils.AverageMeter()
        acc_m = utils.AverageMeter() # Simple Accuracy (Bin prediction)
        mae_m = utils.AverageMeter() # Mean Absolute Error (Scalar Value)
        
        # Pre-calculate bin values for expectation calculation
        # Bin 0 -> min_val, Bin 200 -> max_val
        # shape: (1, 201)
        bin_values = torch.linspace(dataset_train.min_val, dataset_train.max_val, args.num_classes).to(device).unsqueeze(0)
        
        with torch.no_grad():
            for (img_f, img_w, img_s, target) in loader_val:
                img_f, img_w, img_s = img_f.to(device), img_w.to(device), img_s.to(device)
                target = target.to(device)
                
                output = model(img_f, img_w, img_s)
                loss = criterion(output, target)
                
                # 1. Classification Accuracy (Top-1)
                acc1, _ = utils.accuracy(output, target, topk=(1, 5))
                
                # 2. Scalar Value Error (Expected Value vs Ground Truth)
                # Convert Logits -> Probabilities -> Expected Value
                probs = torch.softmax(output, dim=1) # (B, 201)
                pred_val = torch.sum(probs * bin_values, dim=1) # (B,)
                
                # Convert Target Index back to Value
                # Note: This is an approximation if we don't have the original float value, 
                # but consistent with what the model is trying to learn.
                target_val = bin_values[0, target] # (B,)
                
                mae = torch.abs(pred_val - target_val).mean()
                
                val_loss_m.update(loss.item(), img_f.size(0))
                acc_m.update(acc1.item(), img_f.size(0))
                mae_m.update(mae.item(), img_f.size(0))
        
        current_mae = mae_m.avg
        _logger.info(f"Epoch {epoch} Eval: Loss {val_loss_m.avg:.4f}, Bin-Acc {acc_m.avg:.2f}%, MAE {current_mae:.4f}")
        
        # Save Best Checkpoint
        if current_mae < best_mae:
            best_mae = current_mae
            save_path = os.path.join(args.output, f"best_value_net.pth")
            torch.save(model.state_dict(), save_path)
            _logger.info(f"New best model saved with MAE: {best_mae:.4f}")
            
        # Save Latest Checkpoint (Overwrite)
        latest_path = os.path.join(args.output, f"latest_value_net.pth")
        torch.save(model.state_dict(), latest_path)

if __name__ == '__main__':
    main()
