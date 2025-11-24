import os
import argparse
from pathlib import Path
import shutil

def renumber_episodes(target_dir):
    target_path = Path(target_dir)

    if not target_path.exists():
        print(f"Error: Directory {target_path} does not exist.")
        return

    # Find all episode directories
    episode_dirs = sorted([d for d in target_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if not episode_dirs:
        print(f"No episode directories found in {target_path}")
        return

    print(f"Found {len(episode_dirs)} episode directories.")
    
    # Step 1: Sort them by their current index to ensure we preserve the order
    # We extract the number from "episode_XXXX"
    def get_episode_idx(dir_path):
        try:
            return int(dir_path.name.split('_')[-1])
        except ValueError:
            return float('inf') # Put malformed names at the end?

    sorted_episodes = sorted(episode_dirs, key=get_episode_idx)
    
    # Step 2: Rename to temporary names first to avoid collisions 
    # (e.g. renaming episode_0002 to episode_0001 when episode_0001 already exists)
    temp_paths = []
    for i, ep_dir in enumerate(sorted_episodes):
        temp_name = f"temp_renaming_episode_{i:06d}" # Use 6 digits or distinct prefix
        temp_path = target_path / temp_name
        ep_dir.rename(temp_path)
        temp_paths.append(temp_path)
        
    # Step 3: Rename temporary folders to final sequential names
    print("Renaming episodes sequentially...")
    for i, temp_path in enumerate(temp_paths):
        final_name = f"episode_{i:04d}"
        final_path = target_path / final_name
        temp_path.rename(final_path)
        print(f"  {temp_path.name} -> {final_name}")

    print(f"Renumbering complete. {len(temp_paths)} episodes renamed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renumber episode folders sequentially starting from episode_0000.")
    parser.add_argument("target_dir", type=str, help="Path to the directory containing episode_XXXX folders")
    
    args = parser.parse_args()
    
    renumber_episodes(args.target_dir)

