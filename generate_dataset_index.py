import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

def create_dataset_index(input_dir, output_csv):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    data_records = {}
    
    # Categories to look for
    categories = ["success", "failure"]
    
    print("Scanning dataset...")
    
    for category in categories:
        category_dir = input_path / category
        if not category_dir.exists():
            print(f"Warning: Directory {category_dir} not found. Skipping.")
            continue
            
        # Find episode directories
        episode_dirs = sorted([d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        
        for ep_dir in tqdm(episode_dirs, desc=f"Processing {category}"):
            episode_num_str = ep_dir.name.split('_')[-1] # "0000"
            
            # Find all image files
            # The user says images are directly in the episode folder now (based on previous downsample logic),
            # but let's check just in case they are still in 'colors'. 
            # Actually user prompt says "每个subfolder底下有每个frame的三张图片" implying direct containment or in colors?
            # "episode_xxxx (某个数字）的subfolder 每个subfolder底下有每个frame的三张图片" 
            # implies: .../success/episode_0000/000000_color_0.jpg
            
            # Let's support both direct and 'colors' subdir just to be safe/flexible, or assume direct based on description.
            # The description "subfolder底下有...图片" strongly suggests direct children.
            
            image_files = [f for f in ep_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            
            # Use a set to collect unique frame IDs found in this episode
            frame_ids = set()
            
            for img_file in image_files:
                try:
                    # Parse filename: 000135_color_0.jpg
                    frame_str = img_file.name.split('_')[0]
                    frame_num = int(frame_str)
                    frame_ids.add(frame_num)
                except (ValueError, IndexError):
                    pass
            
            if not frame_ids:
                continue
                
            # Calculate total frames for this episode (count of unique frame timestamps)
            total_frames = len(frame_ids)
            
            # Sort frames to process them in order
            sorted_frames = sorted(list(frame_ids))
            
            for frame_num in sorted_frames:
                frame_num_str = f"{frame_num:06d}"
                
                # Key: <flag>_<episode num>_<frame num>
                key = f"{category}_{episode_num_str}_{frame_num_str}"
                
                # Value: [(1) episode num, (2) frame num, (3) flag, (4) total frame number]
                # Note: User requested "episode num" (string or int?), "frame num" (string or int?).
                # Usually int is better for CSV, but let's keep strings for IDs if needed? 
                # Let's use ints for numbers as it's standard for data analysis, or keep string format if user wants exact "0000".
                # The prompt examples imply maintaining the format: "0000", "000000".
                
                record = [
                    episode_num_str,  # (1) episode num (e.g. "0000")
                    frame_num_str,    # (2) frame num (e.g. "000000")
                    category,         # (3) flag ("success" or "failure")
                    total_frames      # (4) total frames
                ]
                
                data_records[key] = record

    # Write to CSV
    print(f"Writing {len(data_records)} records to {output_csv}...")
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header (optional, but good practice)
        writer.writerow(["key", "episode_id", "frame_id", "status", "total_frames"])
        
        # Sort by key to keep CSV ordered
        for key in sorted(data_records.keys()):
            # The user asked for the value list to be converted to CSV.
            # I will write the Key as the first column, then the list items.
            row = [key] + data_records[key]
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV index of all frames in the dataset.")
    parser.add_argument("input_dir", type=str, help="Path to the dataset root (containing success/failure folders)")
    parser.add_argument("output_csv", type=str, help="Path where the output CSV file will be saved")
    
    args = parser.parse_args()
    
    create_dataset_index(args.input_dir, args.output_csv)

