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

def create_dataset_index(input_dir, output_csv, penalty_gap, n_folds, truncate_ratio):
    """
    Generates a CSV index for RECAP Value Function Training.
    
    Key Changes for Pi* / RECAP Reproduction:
    1. Unified Regression: Failure episodes now have a dynamic value slope, shifted down by 'penalty_gap'.
       Value_Fail(t) = Value_Success_Equivalent(t) - Penalty
    2. Data Cleaning: Automatically drops the first N% of failure episodes to remove ambiguity.
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
            
            # Update statistics (Only use Success to define the 'Standard Clock')
            if category == "success":
                if current_length > max_success_length:
                    max_success_length = current_length
            
            episode_metadata.append({
                "category": category,
                "episode_num": episode_num,
                "frame_ids": sorted_frames,
                "total_steps": current_length
            })

    # Fallback
    if max_success_length == 0:
        print("Warning: No successful episodes found. Defaulting max length to 100.")
        max_success_length = 100

    # Define Global Min/Max for Training Script
    # Success Range: [-1.0, 0.0]
    # Failure Range: [-1.0 - penalty, 0.0 - penalty] -> [-2.5, -1.5] (if penalty=1.5)
    # Theoretical Min: -1.0 - penalty_gap
    global_min = -1.0 - penalty_gap
    global_max = 0.0

    print(f"\n--- RECAP Value Logic Stats ---")
    print(f"Max Success Length (T_max): {max_success_length}")
    print(f"Penalty Gap (C_fail): {penalty_gap}")
    print(f"Truncating First {truncate_ratio*100}% of Failure Episodes")
    print(f"Expected Global Value Range: [{global_min:.2f}, {global_max:.2f}]")
    print(f"-----------------\n")

    # --- Sklearn Stratified Group K-Fold ---
    print(f"--- Performing Stratified Group K-Fold ({n_folds} folds) ---")
    
    X_dummy = np.zeros(len(episode_metadata))
    y = []
    groups = []
    
    for meta in episode_metadata:
        label = 1 if meta['category'] == 'success' else 0
        y.append(label)
        unique_group_id = f"{meta['category']}_{meta['episode_num']}"
        groups.append(unique_group_id)

    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    ep_to_fold_map = {}
    
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        for idx in val_idx:
            meta = episode_metadata[idx]
            key = (meta['category'], meta['episode_num'])
            ep_to_fold_map[key] = fold_id

    # --- PASS 2: Generate CSV with Dynamic RECAP Values ---
    data_records = {}
    print("--- Pass 2: Generating records ---")
    
    dropped_frames = 0
    kept_frames = 0
    
    for meta in tqdm(episode_metadata, desc="Processing"):
        category = meta['category']
        episode_num = meta['episode_num']
        sorted_frames = meta['frame_ids']
        total_steps = meta['total_steps']
        
        fold_id = ep_to_fold_map.get((category, episode_num), -1)
        if fold_id == -1: continue
        
        # Determine Start Index (For Data Cleaning)
        start_idx = 0
        if category == "failure" and truncate_ratio > 0:
            # Drop the first N% of frames for failure episodes to remove aliasing
            start_idx = int(total_steps * truncate_ratio)
            dropped_frames += start_idx
        
        for step_idx, frame_num in enumerate(sorted_frames):
            
            # Skip frames if they are in the "Ambiguous Start" zone of a failure
            if step_idx < start_idx:
                continue
            
            kept_frames += 1
            key = f"{category}_{episode_num}_{frame_num}"
            
            # --- CORE RECAP LOGIC ---
            
            # 1. Calculate "Time Component" (Progress)
            # Both Success and Failure calculate "How close to the end?"
            # Success ends at Goal. Failure ends at Drop/Error.
            steps_remaining = total_steps - step_idx
            
            # Normalized time: -1.0 (start) to 0.0 (end)
            # We use max_success_length to standardize the "slope" across all data
            time_val = -1.0 * (steps_remaining / max_success_length)
            
            # Clamp time component to -1.0 (for very long failure episodes)
            time_val = max(-1.0, time_val)
            
            # 2. Calculate Final Value
            if category == "success":
                # Range: [-1.0, 0.0]
                value = time_val
            else:
                # Range: [-2.5, -1.5] (if penalty is 1.5)
                # The slope is the same, but shifted down by penalty
                value = time_val - penalty_gap
            
            record = [
                episode_num,  # Int
                frame_num,    # Int
                category,     # String
                total_steps,  # Int
                f"{value:.5f}", # Float (formatted string)
                fold_id       # Int
            ]
            data_records[key] = record

    # Write to CSV
    print(f"Writing records to {output_csv}...")
    print(f"Stats: Kept {kept_frames} frames. Dropped {dropped_frames} ambiguous failure frames.")
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["key", "episode_id", "frame_id", "status", "total_frames", "normalized_value", "fold"])
        
        for key in sorted(data_records.keys()):
            row = [key] + data_records[key]
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RECAP index with Dynamic Failure Values.")
    parser.add_argument("input_dir", type=str, help="Root directory of dataset")
    parser.add_argument("output_csv", type=str, help="Output CSV path")
    
    # This is the "Gap" between success and failure. 
    # Success range: [-1.0, 0.0]
    # Failure range starts roughly at: -1.0 - 1.5 = -2.5
    parser.add_argument("--penalty", type=float, default=1.5, help="Failure penalty gap (C_fail). Default 1.5")
    
    # IMPORTANT: Data Cleaning
    # Drops the first 30% of failure episodes to prevent confusion
    parser.add_argument("--truncate_failure", type=float, default=0.0, help="Ratio of start frames to drop in failure episodes (0.0 - 1.0). Default 0.3")
    
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds (default: 5)")
    
    args = parser.parse_args()
    
    create_dataset_index(args.input_dir, args.output_csv, args.penalty, args.n_folds, args.truncate_failure)
