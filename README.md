**Read this in other languages: [中文](./docs/README_zh.md).**

|Unitree Robotics  repositories        | link |
|---------------------|------|
| Unitree Datasets   | [unitree datasets](https://huggingface.co/unitreerobotics) |
| AVP Teleoperate    | [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) |


# 0. 📖 Introduction

This repository is used for `lerobot training validation`(Supports LeRobot datasets version 2.0 and above.) and `unitree data conversion`.

`❗Tips： If you have any questions, ideas or suggestions that you want to realize, please feel free to raise them at any time. We will do our best to solve and implement them.`

| Directory          | Description                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | The code in the `lerobot repository` for training;  its corresponding commit version number is `0878c68`.|
| utils         | `unitree data processing tool `   |
| eval_robot    | `unitree real machine inference verification of the model`     |


# 1. 📦 Environment Setup

## 1.1 🦾 LeRobot Environment Setup

The purpose of this project is to use the [LeRobot](https://github.com/huggingface/lerobot) open-source framework to train and test data collected from Unitree robots. Therefore, it is necessary to install the LeRobot-related dependencies first. The installation steps are as follows, and you can also refer to the official [LeRobot](https://github.com/huggingface/lerobot) installation guide:

```bash
# Clone the source code
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git

# If already downloaded:
git submodule update --init --recursive

# Create a conda environment
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# Install LeRobot
cd unitree_lerobot/lerobot && pip install -e .

# Install unitree_lerobot
cd ../../ && pip install -e .
```

## 1.2 🕹️ unitree_sdk2_python

For `DDS communication` on Unitree robots, some dependencies need to be installed. Follow the installation steps below:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 2. ⚙️ Data Collection and Conversion

## 2.1 🖼️ Load Datasets
If you want to directly load the dataset we have already recorded,
Load the [`unitreerobotics/G1_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_ToastedBread_Dataset) dataset from Hugging Face. The default download location is `~/.cache/huggingface/lerobot/unitreerobotics`. If you want to load data from a local source, please change the `root` parameter.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1
dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ToastedBread_Dataset")

from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
```

`visualization`

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/visualize_dataset.py \
    --repo-id unitreerobotics/G1_ToastedBread_Dataset \
    --episode-index 0
```

## 2.2 🔨 Data Collection

If you want to record your own dataset. The open-source teleoperation project [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) can be used to collect data using the Unitree G1 humanoid robot. For more details, please refer to the [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) project.

## 2.3 🛠️ Data Conversion

The data collected using [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) is stored in JSON format. Assuming the collected data is stored in the `$HOME/datasets/task_name`, the format is as follows
```
datasets/                               # Dataset folder
    └── task_name /                     # Task name
        ├── episode_0001                # First trajectory
        │    ├──audios/                 # Audio information
        │    ├──colors/                 # Image information
        │    ├──depths/                 # Depth image information
        │    └──data.json               # State and action information
        ├── episode_0002
        ├── episode_...
        ├── episode_xxx
```

### 2.3.1 🔀 Sort and Rename

When generating datasets for LeRobot, it is recommended to ensure that the data naming convention, starting from `episode_0`, is sequential and continuous. You can use the following script to `sort and rename` the data accordingly.


```bash
python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir $HOME/datasets/task_name
```

#### 2.3.2 🔄 Conversion

Convert `Unitree JSON` Dataset to `LeRobot` Format. You can define your own `robot_type` based on [ROBOT_CONFIGS](https://github.com/unitreerobotics/unitree_IL_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L154).
```bash
# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets \
    --repo-id your_name/repo_task_name \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
```


# 3. 🚀 Training

[For training, please refer to the official LeRobot training example and parameters for further guidance.](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)


- `Train Act Policy`

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=act 
```

- `Train Diffusion Policy`

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=diffusion
```

- `Train Pi0 Policy`

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=pi0
```

# 4. 🤖 Real-World Testing

To test your trained model on a real robot, you can use the eval_g1.py script located in the eval_robot/eval_g1 folder. Here’s how to run it:

[To open the image_server, follow these steps](https://github.com/unitreerobotics/avp_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-server)

```bash
# --policy.path Path to the trained model checkpoint
# --repo_id     Dataset repository ID (Why use it? The first frame state of the dataset is loaded as the initial state)
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_ToastedBread_Dataset

# If you want to evaluate the model's performance on the dataset, use the command below for testing
python unitree_lerobot/eval_robot/eval_g1/eval_g1_dataset.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_ToastedBread_Dataset
```

# 5. 🤔 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Why use `LeRobot v2.0`?** | [Explanation](https://github.com/huggingface/lerobot/pull/461) |
| **401 Client Error: Unauthorized** (`huggingface_hub.errors.HfHubHTTPError`) | Run `huggingface-cli login` to authenticate. |
| **FFmpeg-related errors:**  <br> Q1: `Unknown encoder 'libsvtav1'` <br> Q2: `FileNotFoundError: No such file or directory: 'ffmpeg'` <br> Q3: `RuntimeError: Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed.` | Install FFmpeg: <br> `conda install -c conda-forge ffmpeg` |
| **Access to model `google/paligemma-3b-pt-224` is restricted.** | Run `huggingface-cli login` and request access if needed. |


# 6. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
