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
  /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data \
  /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data/dataset_info.csv

# train value function
python train_value_func.py \
  --csv /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data/dataset_info.csv \
  --data-dir /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data \
  --fold 0 \
  --multiplier 1.5 \
  --num-classes 201 \
  --model efficientnet_b0 \
  --batch-size 32 \
  --epochs 100 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --output /localhome/local-vennw/code/unitree_IL_lerobot/checkpoints/value_func_efnb0_100e