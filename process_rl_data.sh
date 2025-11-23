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
model=efficientnet_b1
epochs=10
batch_size=32
for fold in 0 1 2 3 4; do
  python train_value_func.py \
    --csv /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data/dataset_info.csv \
    --data-dir /localhome/local-vennw/code/datasets/install_trocar_from_tray_realsense_rl_data \
    --fold ${fold} \
    --num-classes 201 \
    --model ${model} \
    --batch-size ${batch_size} \
    --epochs ${epochs} \
    --lr 3e-4 \
    --weight-decay 0.01 \
    --output /localhome/local-vennw/code/unitree_IL_lerobot/checkpoints/value_func_${model}_${epochs}e_fold${fold}_bs${batch_size}
done

