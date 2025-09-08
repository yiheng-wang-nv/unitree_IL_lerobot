# convert unitree json to lerobot format
python unitree_lerobot/utils/sort_and_rename_folders.py \
    --data_dir /localhome/local-vennw/code/data/unitree_recorded_data/wave_two_hands

# then, mkdir a new wave_two_hands folder inside the original wave_two_hands folder

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /localhome/local-vennw/code/data/unitree_recorded_data/wave_two_hands \
    --repo-id venn/unitree_wave_two_hands \
    --robot_type Unitree_G1_Dex3

# sim lab + huggingface data replay
python sim_main.py --device cpu \
  --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129 \
  --action_source=dds

python unitree_lerobot/eval_robot/replay_robot.py \
    --repo_id=venn/unitree_wave_two_hands \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=true \
    --sim=true
