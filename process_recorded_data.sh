if [ "$#" -ne 2 ]; then
echo "Usage: bash process_recorded_data.sh <source-dir> <task-name>"
exit 1
fi

SOURCE_DIR="$1"
TASK_NAME="$2"

# sort and rename success episodes
python unitree_lerobot/utils/sort_and_rename_folders.py \
  --data_dir "${SOURCE_DIR}"

# move episode folders into success/
mkdir -p "${SOURCE_DIR}/success"
find "${SOURCE_DIR}" -maxdepth 1 -type d -name 'episode_*' -exec mv -t "${SOURCE_DIR}/success" {} +

# convert to lerobot format
mkdir -p "${SOURCE_DIR}/${TASK_NAME}"
cp -r "${SOURCE_DIR}/success" "${SOURCE_DIR}/${TASK_NAME}/${TASK_NAME}"

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir "${SOURCE_DIR}/${TASK_NAME}" \
    --repo-id i4h/${TASK_NAME} \
    --robot_type Unitree_G1_Dex3


# example
# cd /home/nvidia/workspace/yiheng/unitree_IL_lerobot
# python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
# --raw-dir ~/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray_realsense_intervention_data_with_relative_pose/interve_data \
# --repo-id i4h/interve_data_demo --robot_type Unitree_G1_Dex3