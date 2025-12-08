# downsample recorded data for value function training
conda activate unitree_lerobot
cd /localhome/local-vennw/code/rl/unitree_IL_lerobot
python clean_data.py \
"/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention" \
"/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop" \
"/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/crop_info.sh"


# intervention data, remove gap
# python detect_action_discontinuity.py \
#     "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop" \
#     --output-csv "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop_trim_info.csv"


# python trim_intervention_data.py \
#     "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop_trim_info.csv" \
#     --input-dir "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop"\
#     --output-dir "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop_no_gap"


conda activate unitree_lerobot
cd /localhome/local-vennw/code/rl/unitree_IL_lerobot

python downsample_data.py \
    --interval=3 \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/success_crop" \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/success_crop_downsampled"

python downsample_data.py \
    --interval=3 \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop" \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/intervention_crop_downsampled"

python downsample_data.py \
    --interval=3 \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/failure" \
    "/localhome/local-vennw/code/rl/datasets/trocar_post_train_data/failure_downsampled"

# /home/nvidia/workspace/datasets/install_trocar_from_tray_realsense_rl_data

python generate_dataset_index.py \
  /localhome/local-vennw/code/rl/datasets/trocar_post_train_data \
  /localhome/local-vennw/code/rl/datasets/trocar_post_train_data/dataset_info.csv \
  --success success_crop_downsampled --failure failure_downsampled --intervention intervention_crop_downsampled --truncate_failure 0.1

# train value function
model=efficientnet_b1
epochs=10
batch_size=32
for fold in 0 1 2 3 4; do
  python train_value_func.py \
    --csv /localhome/local-vennw/code/rl/datasets/trocar_post_train_data/dataset_info.csv\
    --data-dir /localhome/local-vennw/code/rl/datasets/trocar_post_train_data \
    --fold ${fold} \
    --num-classes 201 \
    --model ${model} \
    --batch-size ${batch_size} \
    --epochs ${epochs} \
    --lr 3e-4 \
    --weight-decay 0.01 \
    --output /localhome/local-vennw/code/rl/checkpoints/value_func_${model}_${epochs}e_fold${fold}_bs${batch_size}
done

