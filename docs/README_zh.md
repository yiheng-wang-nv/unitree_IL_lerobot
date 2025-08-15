|Unitree Robotics  repositories        | link |
|---------------------|------|
| Unitree Datasets   | [unitree datasets](https://huggingface.co/unitreerobotics) |
| AVP Teleoperate    | [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) |


# 0. 📖 介绍

此存储库是使用`lerobot训练验证`(支持lerobot 数据集 v2.0以上版本)和`unitree数据转换`

`❗Tips：如果您有任何疑问，想法或建议，请随时随时提出它们。我们将尽最大努力解决和实现。`

| 目录          | 说明                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | `lerobot` 仓库代码，其对应的 commit 版本号为 `0878c68` |
| utils         | `unitree 数据处理工具`     |
| eval_robot    | `unitree 模型真机推理验证`     |

# 1. 📦 环境安装

## 1.1 🦾 LeRobot 环境安装

本项的目的是使用[LeRobot](https://github.com/huggingface/lerobot)开源框架训练并测试基于 Unitree 机器人采集的数据。所以首先需要安装 LeRobot 相关依赖。安装步骤如下，也可以参考[LeRobot](https://github.com/huggingface/lerobot)官方进行安装:


```bash
# 下载源码
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git

# 已经下载:
git submodule update --init --recursive

# 创建 conda 环境
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# 安装 LeRobot
cd lerobot && pip install -e .

# 安装 unitree_lerobot
cd .. && pip install -e .
```


## 1.2 🕹️ unitree_sdk2_python
针对 Unitree 机器人`dds通讯`需要安装一些依赖,安装步骤如下:
```
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 2. ⚙️ 数据采集与转换

## 2.1 🖼️ 数据加载测试
如果你想加载我们已经录制好的数据集, 你可以从 huggingface上加载 [`unitreerobotics/G1_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_ToastedBread_Dataset) 数据集, 默认下载到`~/.cache/huggingface/lerobot/unitreerobotics`. 如果想从加载本地数据请更改 `root` 参数 

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

`可视化` 

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/visualize_dataset.py \
    --repo-id unitreerobotics/G1_ToastedBread_Dataset \
    --episode-index 0
```

## 2.2 🔨 数据采集

如果你想录制自己的数据集, 可以使用开源的遥操作项目[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) 对 Unitree G1 人形机器人进行数据采集，具体可参考[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)项目。

## 2.3 🛠️ 数据转换

使用[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)采集的数据是采用 JSON 格式进行存储。假如采集的数据存放在`$HOME/datasets/task_name` 目录中，格式如下:
```
datasets/                               # 数据集文件夹
    └── task_name /                     # 任务名称
        ├── episode_0001                # 第一条轨迹
        │    ├──audios/                 # 声音信息
        │    ├──colors/                 # 图像信息
        │    ├──depths/                 # 深度图像信息
        │    └──data.json               # 状态以及动作信息
        ├── episode_0002
        ├── episode_...
        ├── episode_xxx
```

### 2.3.1 🔀 排序和重命名

生成 lerobot 的数据集时，最好保证数据的`episode_0`命名是从 0 开始且是连续的，使用下面脚本对数据进行排序处理

```bash
python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir $HOME/datasets/task_name
```

### 2.3.2 🔄 转换

转换`json`格式到`lerobot`格式，你可以根据 [ROBOT_CONFIGS](https://github.com/unitreerobotics/unitree_IL_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L154) 去定义自己的 `robot_type`

```bash
# --raw-dir     对应json的数据集目录
# --repo-id     对应自己的repo-id 
# --push_to_hub 是否上传到云端 
# --robot_type  对应的机器人类型 

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py  
    --raw-dir $HOME/datasets    
    --repo-id your_name/repo_task_name  
    --robot_type Unitree_G1_Dex3    # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    --push_to_hub
```


# 3. 🚀 训练

[请详细阅读官方lerobot训练实例与相关参数](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)


- `训练 act`
```
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=act 
```

- `训练 Diffusion Policy`
```
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=diffusion
```
- `训练 pi0`
```
cd unitree_lerobot/lerobot

python src/lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=pi0
```

# 4. 🤖 真机测试
[如何打开 image_server](https://github.com/unitreerobotics/avp_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-server)
```bash
# --policy.path 训练保存模型路径
# --repo_id     训练加载的数据集(为什么要用? 加载数据集中第一帧状态做为起始动作)

python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_ToastedBread_Dataset


# 如果你想验证模型在数据集上的表现 使用下面去测试
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


# 6. 🙏 致谢

此代码基于以下开源代码库进行构建。请访问以下链接查看相关的许可证：

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
