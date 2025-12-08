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

def create_dataset_index(input_dir, output_csv, penalty_gap, n_folds, truncate_ratio, 
                         success_folder, failure_folder, intervention_folder):
    """
    Generates a CSV index for RECAP Value Function Training.
    
    Key Features:
    1. Unified Regression Target: 
       - Success/Intervention: Linear ramp from -1.0 to 0.0.
       - Failure: Linear ramp shifted down by 'penalty_gap' (e.g., -2.5 to -1.5).
    2. Data Cleaning: 
       - Automatically drops the first N% of failure episodes to remove perceptual aliasing (ambiguous start states).
    3. Optional Intervention:
       - Can optionally include 'intervention' data treated as high-value success samples.
    
    Args:
        success_folder: Folder name for success episodes (None to skip)
        failure_folder: Folder name for failure episodes (None to skip)
        intervention_folder: Folder name for intervention episodes (None to skip)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Build category mapping: category_type -> folder_name
    # category_type is used internally for logic (success/failure/intervention)
    # folder_name is the actual directory name
    category_folders = {}
    if success_folder:
        category_folders["success"] = success_folder
    if failure_folder:
        category_folders["failure"] = failure_folder
    if intervention_folder:
        category_folders["intervention"] = intervention_folder
    
    if not category_folders:
        print("Error: At least one folder must be specified (--success, --failure, or --intervention)")
        return
    
    categories = list(category_folders.keys())
    
    episode_metadata = []
    max_success_length = 0 # Baseline for normalizing time steps
    
    print(f"--- Pass 1: Scanning dataset at {input_path} ---")
    print(f"Category -> Folder mapping:")
    for cat, folder in category_folders.items():
        print(f"  {cat}: {folder}")
    
    # --- PASS 1: Scan directories ---
    for category in categories:
        folder_name = category_folders[category]
        category_dir = input_path / folder_name
        if not category_dir.exists():
            print(f"Warning: Directory {category_dir} not found. Skipping '{category}'.")
            continue
            
        # List all episode directories
        episode_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        
        for ep_dir in tqdm(episode_dirs, desc=f"Scanning {category}"):
            try:
                # Parse episode ID
                episode_num = int(ep_dir.name.split('_')[-1])
            except ValueError:
                continue
            
            # Find image files
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
            
            # Update Max Length Statistics
            # Both 'success' and 'intervention' contribute to defining the standard task duration
            if category in ["success", "intervention"]:
                if current_length > max_success_length:
                    max_success_length = current_length
            
            episode_metadata.append({
                "category": category,
                "episode_num": episode_num,
                "frame_ids": sorted_frames,
                "total_steps": current_length
            })

    # Fallback to prevent division by zero
    if max_success_length == 0:
        print("Warning: No successful/intervention episodes found. Defaulting max length to 100.")
        max_success_length = 100

    print(f"\n--- RECAP Value Logic Stats ---")
    print(f"Max Standard Length (T_max): {max_success_length}")
    print(f"Penalty Gap (C_fail): {penalty_gap}")
    if "failure" in categories:
        print(f"Truncating First {truncate_ratio*100}% of FAILURE episodes to remove aliasing.")
    if "intervention" in categories:
        print(f"Intervention episodes are INCLUDED and treated as SUCCESS (High Value).")
    print(f"-----------------\n")

    # --- Stratified Group K-Fold ---
    # We use StratifiedGroupKFold to ensure:
    # 1. Frames from the same episode stay in the same fold (Group constraint).
    # 2. Each fold has a balanced ratio of success/failure (Stratified constraint).
    
    X_dummy = np.zeros(len(episode_metadata))
    y = [] # Binary label for stratification
    groups = []
    
    for meta in episode_metadata:
        # y: success/intervention = 1, failure = 0
        if meta['category'] in ["success", "intervention"]:
            label = 1
        else:
            label = 0
        y.append(label)
        
        # Unique Group ID to prevent data leakage
        unique_group_id = f"{meta['category']}_{meta['episode_num']}"
        groups.append(unique_group_id)

    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    ep_to_fold_map = {}
    
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        for idx in val_idx:
            meta = episode_metadata[idx]
            key = (meta['category'], meta['episode_num'])
            ep_to_fold_map[key] = fold_id

    # --- PASS 2: Generate CSV Records ---
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
        
        # --- Data Cleaning (Truncation) ---
        start_idx = 0
        if category == "failure" and truncate_ratio > 0:
            # Drop the first N% of frames for failure episodes.
            # This removes the "normal-looking" start of failed episodes.
            start_idx = int(total_steps * truncate_ratio)
            dropped_frames += start_idx
        
        for step_idx, frame_num in enumerate(sorted_frames):
            
            # Skip truncated frames
            if step_idx < start_idx:
                continue
            
            kept_frames += 1
            key = f"{category}_{episode_num}_{frame_num}"
            
            # --- Value Calculation ---
            # Calculate remaining steps
            steps_remaining = total_steps - step_idx
            
            # Normalized Time: -1.0 (start) -> 0.0 (end)
            # We use max_success_length to standardize the slope
            time_val = -1.0 * (steps_remaining / max_success_length)
            time_val = max(-1.0, time_val) # Clamp start value
            
            if category in ["success", "intervention"]:
                # Success/Intervention Range: [-1.0, 0.0]
                value = time_val
            else:
                # Failure Range: Shifted down by penalty
                # e.g., if penalty is 1.5, range becomes [-2.5, -1.5]
                value = time_val - penalty_gap
            
            record = [
                episode_num,  
                frame_num,    
                category,     
                total_steps,  
                f"{value:.5f}", 
                fold_id       
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
    parser = argparse.ArgumentParser(
        description="Generate RECAP dataset index with unified value regression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage with default folder names
  python generate_dataset_index.py /path/to/data output.csv --success success --failure failure
  
  # Include intervention data
  python generate_dataset_index.py /path/to/data output.csv --success success --failure failure --intervention intervention
  
  # Custom folder names
  python generate_dataset_index.py /path/to/data output.csv --success good_demos --failure bad_demos
  
  # Only success data (no failure)
  python generate_dataset_index.py /path/to/data output.csv --success success
        """
    )
    
    parser.add_argument("input_dir", type=str, help="Root directory of the dataset")
    parser.add_argument("output_csv", type=str, help="Path to save the generated CSV")
    
    # Folder Names (at least one required)
    folder_group = parser.add_argument_group('Data Folders', 'Specify folder names for each category (at least one required)')
    folder_group.add_argument("--success", type=str, default=None, metavar="FOLDER",
                              help="Folder name for SUCCESS episodes (e.g., 'success')")
    folder_group.add_argument("--failure", type=str, default=None, metavar="FOLDER",
                              help="Folder name for FAILURE episodes (e.g., 'failure')")
    folder_group.add_argument("--intervention", type=str, default=None, metavar="FOLDER",
                              help="Folder name for INTERVENTION episodes (treated as success)")
    
    # RECAP Logic Parameters
    parser.add_argument("--penalty", type=float, default=1.5, 
                        help="Value penalty gap for failure episodes (default: 1.5)")
    parser.add_argument("--truncate_failure", type=float, default=0.3, 
                        help="Ratio of start frames to drop in failure episodes (0.0-1.0, default: 0.3)")
    
    # Validation Setup
    parser.add_argument("--n_folds", type=int, default=5, 
                        help="Number of folds for Cross-Validation (default: 5)")
    
    args = parser.parse_args()
    
    # Validate at least one folder is specified
    if not any([args.success, args.failure, args.intervention]):
        parser.error("At least one folder must be specified: --success, --failure, or --intervention")
    
    create_dataset_index(
        args.input_dir, 
        args.output_csv, 
        args.penalty, 
        args.n_folds, 
        args.truncate_failure,
        args.success,
        args.failure,
        args.intervention
    )
