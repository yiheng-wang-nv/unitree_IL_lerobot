import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_crop_info(crop_info_path):
    """Parses the crop_info.sh file into a dictionary {episode_id: last_frame_index}."""
    crop_map = {}
    with open(crop_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                ep_idx_str, last_frame_str = line.split(',')
                crop_map[int(ep_idx_str)] = int(last_frame_str)
            except ValueError:
                print(f"Warning: Skipping invalid line in crop info: {line}")
    return crop_map

def clean_dataset(input_dir, output_dir, crop_info_path):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return
    
    if not Path(crop_info_path).exists():
         print(f"Error: Crop info file {crop_info_path} does not exist.")
         return

    # Parse crop info
    crop_map = parse_crop_info(crop_info_path)
    print(f"Loaded crop info for {len(crop_map)} episodes.")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find episode directories
    episode_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if not episode_dirs:
        print(f"No episode directories found in {input_path}")
        return

    for ep_dir in tqdm(episode_dirs, desc="Processing Episodes"):
        # Extract episode index from name "episode_XXXX"
        try:
            ep_idx = int(ep_dir.name.split('_')[-1])
        except ValueError:
            print(f"Skipping folder with invalid format: {ep_dir.name}")
            continue

        # Determine cutoff frame
        last_frame = crop_map.get(ep_idx)
        
        # Setup output episode path
        out_ep_dir = output_path / ep_dir.name
        out_colors_dir = out_ep_dir / "colors"
        out_ep_dir.mkdir(parents=True, exist_ok=True)
        out_colors_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Process Images
        colors_dir = ep_dir / "colors"
        if colors_dir.exists():
            for img_file in colors_dir.iterdir():
                if img_file.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                    continue
                
                try:
                    # Parse frame ID from filename "000000_color_0.jpg"
                    frame_id = int(img_file.name.split('_')[0])
                    
                    # Copy only if within range (or if no limit specified for this episode)
                    if last_frame is None or frame_id <= last_frame:
                        shutil.copy2(img_file, out_colors_dir / img_file.name)
                except ValueError:
                    pass # Skip files that don't match pattern

        # 2. Process data.json
        data_json_path = ep_dir / "data.json"
        if data_json_path.exists():
            with open(data_json_path, 'r') as f:
                data = json.load(f)
            
            # Filter the "data" list
            if "data" in data and isinstance(data["data"], list):
                original_len = len(data["data"])
                if last_frame is not None:
                    # Filter items where "idx" <= last_frame
                    # Assuming "idx" corresponds to the frame ID
                    data["data"] = [item for item in data["data"] if item.get("idx", float('inf')) <= last_frame]
                
                # Write processed json
                with open(out_ep_dir / "data.json", 'w') as f:
                    json.dump(data, f, indent=4)
        
        # Copy metadata.json if exists (no filtering needed usually, or update frame count?)
        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
             shutil.copy2(meta_path, out_ep_dir / "metadata.json")

        # Copy any other files/folders in the episode directory to maintain structure
        for item in ep_dir.iterdir():
            # Skip items we've already handled or don't want to blindly copy
            if item.name in ["colors", "data.json", "metadata.json"]:
                continue
            
            dest = out_ep_dir / item.name
            if item.is_dir():
                # If it's a directory (like depths/audios etc), copy recursively
                if not dest.exists():
                    shutil.copytree(item, dest)
            else:
                # If it's a file (like label.json or others), copy it
                shutil.copy2(item, dest)

    print(f"Cleaning complete. Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop robot rollout episodes based on a cutoff frame file.")
    parser.add_argument("input_dir", type=str, help="Path to input dataset directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    parser.add_argument("crop_info", type=str, help="Path to .sh file containing 'episode_id,last_frame'")
    
    args = parser.parse_args()
    
    clean_dataset(args.input_dir, args.output_dir, args.crop_info)

