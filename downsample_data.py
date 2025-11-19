import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def downsample_dataset(input_dir, output_dir, interval=3):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all episode directories
    episode_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if not episode_dirs:
        print(f"No episode directories found in {input_path}")
        return

    print(f"Found {len(episode_dirs)} episodes. Processing...")

    for ep_dir in tqdm(episode_dirs, desc="Processing Episodes"):
        # Create corresponding output episode structure
        rel_path = ep_dir.relative_to(input_path)
        out_ep_dir = output_path / rel_path
        out_colors_dir = out_ep_dir / "colors"
        
        out_colors_dir.mkdir(parents=True, exist_ok=True)

        # Copy metadata.json if it exists
        meta_file = ep_dir / "metadata.json"
        if meta_file.exists():
            shutil.copy2(meta_file, out_ep_dir / "metadata.json")

        # Process images in 'colors' directory
        colors_dir = ep_dir / "colors"
        if not colors_dir.exists():
            continue

        # Gather all image files
        image_files = sorted([f for f in colors_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
        
        # Group images by frame ID (assuming format XXXXXX_color_Y.jpg)
        # We want to keep frames 0, 3, 6, ... (indices divisible by interval)
        
        # It's safer to rely on the frame ID in the filename rather than just file sorting order
        # if we want to be robust, but assuming standard naming conventions:
        
        for img_file in image_files:
            try:
                # Parse frame ID from filename "000000_color_0.jpg" -> 0
                frame_id_str = img_file.name.split('_')[0]
                frame_id = int(frame_id_str)
                
                if frame_id % interval == 0:
                    shutil.copy2(img_file, out_colors_dir / img_file.name)
            except ValueError:
                # If filename format doesn't match expected pattern, just skip or copy? 
                # Let's skip and warn to be safe.
                # print(f"Skipping file with unexpected name format: {img_file.name}")
                pass

    print(f"Downsampling complete. Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample robot rollout data by keeping every Nth frame.")
    parser.add_argument("input_dir", type=str, help="Path to the input dataset directory containing episode_XXXX folders")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--interval", type=int, default=3, help="Downsample interval (default: 3, keeps 1/3 frames)")
    
    args = parser.parse_args()
    
    downsample_dataset(args.input_dir, args.output_dir, args.interval)

