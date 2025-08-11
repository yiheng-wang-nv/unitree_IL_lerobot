''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
'''

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from multiprocessing import Array, Lock

from lerobot.policies.factory import make_policy
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            # if "images" in name:
            #     observation[name] = observation[name].type(torch.float32) / 255
            #     observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()

    # send_real_robot: If you want to read observations from the dataset and send them to the real robot, set this to True.  
    # (This helps verify whether the model has generalization ability to the environment, as there are inevitably differences between the real environment and the training data environment.)
    robot_config = {
        'arm_type': 'g1',
        'hand_type': "dex3",
        'send_real_robot': False,
    }

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]
    to_idx = dataset.episode_data_index["to"][0].item()

    camera_names = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]
    ground_truth_actions = []
    predicted_actions = []

    if robot_config['send_real_robot']:
        # arm
        arm_ctrl = G1_29_ArmController()
        init_left_arm_pose = step['observation.state'][:14].cpu().numpy()

        # hand
        if robot_config['hand_type'] == "dex3":
            left_hand_array = Array('d', 7, lock = True)          # [input]
            right_hand_array = Array('d', 7, lock = True)         # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
            init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
            init_right_hand_pose = step['observation.state'][21:].cpu().numpy()

        elif robot_config['hand_type'] == "gripper":
            left_hand_array = Array('d', 1, lock=True)             # [input]
            right_hand_array = Array('d', 1, lock=True)            # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
            init_left_hand_pose = step['observation.state'][14].cpu().numpy()
            init_right_hand_pose = step['observation.state'][15].cpu().numpy()
        else:
            pass

    #===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':
        
        if robot_config['send_real_robot']:
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            print("init robot pose")
            arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
            left_hand_array[:] = init_left_hand_pose
            right_hand_array[:] = init_right_hand_pose

            print("wait robot to pose")
            time.sleep(1)

        frequency = 50.0

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            step = dataset[step_idx]
            observation = {}

            for cam_name in camera_names:
                observation[f"observation.images.{cam_name}"] = step[f"observation.images.{cam_name}"]
            observation["observation.state"] = step["observation.state"]

            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )

            action = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action)

            if robot_config['send_real_robot']:
                # exec action
                arm_ctrl.ctrl_dual_arm(action[:14], np.zeros(14))
                if robot_config['hand_type'] == "dex3":
                    left_hand_array[:] = action[14:21]
                    right_hand_array[:] = action[21:]
                elif robot_config['hand_type'] == "gripper":
                    left_hand_array[:] = action[14]
                    right_hand_array[:] = action[15]
            
                time.sleep(1/frequency)
        
        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)

        # Get the number of timesteps and action dimensions
        n_timesteps, n_dims = ground_truth_actions.shape

        # Create a figure with subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
        fig.suptitle('Ground Truth vs Predicted Actions')

        # Plot each dimension
        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label='Ground Truth', color='blue')
            ax.plot(predicted_actions[:, i], label='Predicted', color='red', linestyle='--')
            ax.set_ylabel(f'Dim {i+1}')
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel('Timestep')

        plt.tight_layout()
        # plt.show()

        time.sleep(1)
        plt.savefig('figure.png')


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id = cfg.repo_id)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
