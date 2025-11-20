import os
import csv
import argparse
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import scikit-learn for robust splitting
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    print("Error: scikit-learn is required. Please install it via 'pip install scikit-learn'.")
    exit(1)

def create_dataset_index(input_dir, output_csv, failure_multiplier, n_folds):
    """
    Generates a CSV index for Recap Value Function Training.
    
    Implements the reward definition from Section V-C of the pi* paper:
    1. Finds Max Task Length (T_max) from successful episodes.
    2. Normalizes Success values to [-1, 0].
    3. Sets Failure values to -1 * failure_multiplier (e.g., -1.5).
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Store metadata
    episode_metadata = []
    
    # Track max length of successful episodes for Normalization
    max_success_length = 0
    
    categories = ["success", "failure"]
    
    print(f"--- Pass 1: Scanning dataset at {input_path} ---")
    
    # --- PASS 1: Scan directories ---
    for category in categories:
        category_dir = input_path / category
        if not category_dir.exists():
            print(f"Warning: Directory {category_dir} not found. Skipping.")
            continue
            
        episode_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        
        for ep_dir in tqdm(episode_dirs, desc=f"Scanning {category}"):
            try:
                # Extract episode number as integer (e.g., "episode_0000" -> 0)
                episode_num = int(ep_dir.name.split('_')[-1])
            except ValueError:
                continue
            
            # Find all image files
            image_files = [f for f in ep_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            
            # Extract unique frame IDs
            frame_ids = set()
            for img_file in image_files:
                try:
                    frame_str = img_file.name.split('_')[0]
                    frame_num = int(frame_str)
                    frame_ids.add(frame_num)
                except (ValueError, IndexError):
                    pass
            
            if not frame_ids:
                continue
                
            sorted_frames = sorted(list(frame_ids))
            current_length = len(sorted_frames)
            
            # Update statistics
            if category == "success":
                if current_length > max_success_length:
                    max_success_length = current_length
            
            episode_metadata.append({
                "category": category,
                "episode_num": episode_num,
                "frame_ids": sorted_frames,
                "total_steps": current_length
            })

    # Fallback if no success episodes are found
    if max_success_length == 0:
        print("Warning: No successful episodes found. Defaulting max length to 100.")
        max_success_length = 100

    print(f"\n--- Normalization Stats (Paper Section V-C) ---")
    print(f"Max Success Length (T_max): {max_success_length}")
    print(f"Success Value Range: [-1.0, 0.0]")
    print(f"Failure Value Constant: {-failure_multiplier}")
    print(f"-----------------\n")

    # --- Sklearn Stratified Group K-Fold ---
    print(f"--- Performing Stratified Group K-Fold ({n_folds} folds) ---")
    
    X_dummy = np.zeros(len(episode_metadata))
    y = []
    groups = []
    
    for meta in episode_metadata:
        # y: success=1, failure=0
        label = 1 if meta['category'] == 'success' else 0
        y.append(label)
        
        # Unique group ID
        unique_group_id = f"{meta['category']}_{meta['episode_num']}"
        groups.append(unique_group_id)

    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    ep_to_fold_map = {}
    
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        for idx in val_idx:
            meta = episode_metadata[idx]
            key = (meta['category'], meta['episode_num'])
            ep_to_fold_map[key] = fold_id

    # --- PASS 2: Generate CSV with Normalized Values ---
    data_records = {}
    print("--- Pass 2: Generating records ---")
    
    for meta in tqdm(episode_metadata, desc="Processing"):
        category = meta['category']
        episode_num = meta['episode_num']
        sorted_frames = meta['frame_ids']
        total_steps = meta['total_steps']
        
        fold_id = ep_to_fold_map.get((category, episode_num), -1)
        if fold_id == -1: continue
        
        for step_idx, frame_num in enumerate(sorted_frames):
            
            key = f"{category}_{episode_num}_{frame_num}"
            
            # --- NORMALIZATION LOGIC ---
            # Success: -(Remaining Steps) / Max Length
            # Range: -1.0 (start) to 0.0 (end)
            if category == "success":
                steps_remaining = total_steps - step_idx
                value = -1.0 * (steps_remaining / max_success_length)
                
                # Clamp value to be safe (shouldn't happen if math is right, but just in case)
                value = max(-1.0, value)
            
            # Failure: Fixed Penalty Multiplier
            # Range: e.g., -1.5
            else:
                value = -1.0 * failure_multiplier
            
            record = [
                episode_num,  # Int
                frame_num,    # Int
                category,     # String
                total_steps,  # Int
                f"{value:.4f}", # Float (formatted string)
                fold_id       # Int
            ]
            data_records[key] = record

    # Write to CSV
    print(f"Writing records to {output_csv}...")
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # normalized_value stores the float value (Ground Truth R)
        writer.writerow(["key", "episode_id", "frame_id", "status", "total_frames", "normalized_value", "fold"])
        
        for key in sorted(data_records.keys()):
            row = [key] + data_records[key]
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Recap index with Normalized Values.")
    parser.add_argument("input_dir", type=str, help="Root directory of dataset")
    parser.add_argument("output_csv", type=str, help="Output CSV path")
    
    # Paper implies a failure penalty significantly lower than -1.0.
    # 1.5 is a good default (-1.5 vs -1.0 gap).
    parser.add_argument("--multiplier", type=float, default=1.5, help="Failure penalty multiplier (default: 1.5)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds (default: 5)")
    
    args = parser.parse_args()
    
    create_dataset_index(args.input_dir, args.output_csv, args.multiplier, args.n_folds)
