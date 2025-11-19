# downsample recorded data for value function training
conda activate unitree_lerobot
cd /home/nvidia/workspace/yiheng/unitree_IL_lerobot
python clean_data.py \
"/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense/success/" \
"/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense/success_crop/" \
"/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense/crop_info.sh"


conda activate unitree_lerobot
cd /home/nvidia/workspace/yiheng/unitree_IL_lerobot

python downsample_data.py \
    --interval=3 \
    "/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense/success_crop/" \
    "/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense/success_crop_downsampled/"

python downsample_data.py \
    --interval=3 \
    "/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense_rollout/" \
    "/home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense_rollout_downsampled/"

# /home/nvidia/workspace/datasets/install_trocar_from_tray_realsense_rl_data

python generate_dataset_index.py \
  /home/nvidia/workspace/datasets/install_trocar_from_tray_realsense_rl_data \
  /home/nvidia/workspace/datasets/install_trocar_from_tray_realsense_rl_data/dataset_info.csv