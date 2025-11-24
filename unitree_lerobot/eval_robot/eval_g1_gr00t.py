"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging
import sys
import select

import numpy as np
from pprint import pformat
from dataclasses import asdict

from lerobot.utils.utils import (
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from multiprocessing.sharedctypes import SynchronizedArray

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations_gr00t,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import UnitreeG1DataConfig_v5

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    policy,
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
):

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()
    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)
        
        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof, sim_state_subscriber = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof", "sim_state_subscriber"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )
        left_ee_init = None
        right_ee_init = None
        # Get initial pose from the first step of the dataset
        from_idx = dataset.episode_data_index["from"][cfg.episodes].item()
        print(f"from_idx: {dataset[from_idx]}")
        step = dataset[from_idx]
        print(f"step: {step}")
        # init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        # Derive initial end-effector targets from dataset if available
        try:
            if cfg.ee:
                full_step_state = step["observation.state"].cpu().numpy()
                if full_step_state.shape[0] >= (arm_dof + 2 * ee_dof):
                    left_ee_init = full_step_state[arm_dof : arm_dof + ee_dof]
                    right_ee_init = full_step_state[arm_dof + ee_dof : arm_dof + 2 * ee_dof]
                else:
                    # Fallback to zeros if dataset does not include ee states
                    left_ee_init = np.zeros((ee_dof,), dtype=float)
                    right_ee_init = np.zeros((ee_dof,), dtype=float)
        except Exception:
            # Conservative fallback
            if cfg.ee:
                left_ee_init = np.zeros((ee_dof,), dtype=float)
                right_ee_init = np.zeros((ee_dof,), dtype=float)
        logger_mp.info("Initializing robot to starting pose...")
        tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
        robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1.0)  # Give time for the robot to move

        quit_program = False
        while not quit_program:
            user_input = input("Press 's' to start evaluation, 'r' to reset to initial pose, or 'q' to quit: ")
            user_input = user_input.strip().lower()
            print(f"user_input: {user_input}")

            if user_input == "q":
                logger_mp.info("Quit requested. Exiting.")
                break

            if user_input == "r":
                logger_mp.info("Resetting to initial pose...")
                tau = arm_ik.solve_tau(init_arm_pose)
                arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
                # Also reset Dex (end-effector) targets to initial dataset pose if configured
                if cfg.ee and left_ee_init is not None and right_ee_init is not None:
                    try:
                        with ee_shared_mem["lock"]:
                            if isinstance(ee_shared_mem["left"], SynchronizedArray):
                                ee_shared_mem["left"][:] = to_list(left_ee_init)
                                ee_shared_mem["right"][:] = to_list(right_ee_init)
                            elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                                # Scalar case (e.g., simple gripper)
                                ee_shared_mem["left"].value = to_scalar(left_ee_init)
                                ee_shared_mem["right"].value = to_scalar(right_ee_init)
                    except Exception:
                        pass
                time.sleep(1.0)
                continue

            if user_input != "s":
                logger_mp.info("Unrecognized input. Use 's' to start, 'r' to reset, 'q' to quit.")
                continue

            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz. Press 'r'+Enter to stop & reset; 'q'+Enter to quit.")
            idx = 0
            while True:
                # Non-blocking command check (press key then Enter)
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == "r":
                            logger_mp.info("Stop & reset requested.")
                            break
                        if cmd == "q":
                            logger_mp.info("Quit requested during evaluation loop.")
                            quit_program = True
                            break
                except Exception:
                    pass

                loop_start_time = time.perf_counter()

                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations_gr00t(
                    tv_img_array, wrist_img_array, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                state_tensor = torch.from_numpy(np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)).float()
                # observation["observation.state"] = state_tensor
                observation["state.left_arm"] = state_tensor[:7]
                observation["state.right_arm"] = state_tensor[7:14]
                observation["state.left_hand"] = state_tensor[14:21]
                observation["state.right_hand"] = state_tensor[21:28]
                # 2. Get Action from Policy
                actions = predict_action(
                    observation,
                    policy,
                    step["task"],
                    use_dataset=False,
                    use_gr00t=True,
                )

                # 3. Execute Action
                for action_np in actions:
                    loop_start_time = time.perf_counter()
                    arm_action = action_np[:arm_dof]
                    tau = arm_ik.solve_tau(arm_action)
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)

                    if cfg.ee:
                        ee_action_start_idx = arm_dof
                        left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                        right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_action)
                            ee_shared_mem["right"][:] = to_list(right_ee_action)
                        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["left"].value = to_scalar(left_ee_action)
                            ee_shared_mem["right"].value = to_scalar(right_ee_action)

                    if cfg.visualization:
                        visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                    idx += 1
                    # Maintain frequency
                    time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

            if quit_program:
                break

            # After a session ends with 'r', return to initial pose automatically
            logger_mp.info("Returning to initial pose...")
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            # Also reset Dex (end-effector) targets after each session ends with 'r'
            if cfg.ee and left_ee_init is not None and right_ee_init is not None:
                try:
                    with ee_shared_mem["lock"]:
                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_init)
                            ee_shared_mem["right"][:] = to_list(right_ee_init)
                        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["left"].value = to_scalar(left_ee_init)
                            ee_shared_mem["right"].value = to_scalar(right_ee_init)
                except Exception:
                    pass
            time.sleep(1.0)

    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)
        # Clean up sim state subscriber if it exists
        if 'sim_state_subscriber' in locals() and sim_state_subscriber:
            sim_state_subscriber.stop_subscribe()
            logger_mp.info("SimStateSubscriber cleaned up")


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    data_config = UnitreeG1DataConfig_v5()
    # data_config = UnitreeG1DataConfig_v4()
    # data_config = UnitreeG1DataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=cfg.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        device="cuda",
    )
    with torch.no_grad():
        eval_policy(policy, cfg=cfg, dataset=dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
