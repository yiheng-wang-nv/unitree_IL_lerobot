if [ "$#" -ne 2 ]; then
echo "Usage: bash process_recorded_data.sh <source-dir> <task-name>"
exit 1
fi

SOURCE_DIR="$1"
TASK_NAME="$2"

conda activate unitree_lerobot

# put episodes into success/failure/unspecified folders
python unitree_lerobot/utils/organize_episodes_by_label.py \
  "${SOURCE_DIR}"

# sort and rename success episodes
python unitree_lerobot/utils/sort_and_rename_folders.py \
  --data_dir "${SOURCE_DIR}/success"

# convert to lerobot format
mkdir -p "${SOURCE_DIR}/${TASK_NAME}"
cp -r "${SOURCE_DIR}/success" "${SOURCE_DIR}/${TASK_NAME}"

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir "${SOURCE_DIR}" \
    --repo-id i4h/${TASK_NAME} \
    --robot_type Unitree_G1_Dex3
