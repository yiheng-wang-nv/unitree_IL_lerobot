"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import sys
import importlib.util
import logging
import select

import torch
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
    to_list,
    to_scalar,
    EvalRealConfig
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def replay_trajectory(arm_ctrl, arm_ik, ee_shared_mem, traj_data, reverse=False, speed=1.0):
    """Replay a recorded safe trajectory. If reverse=True, play it backwards."""
    arm_q_seq = traj_data["arm_q"]
    hand_seq = traj_data["hand_state"] if "hand_state" in traj_data else None
    freq = float(traj_data["frequency"]) * speed

    indices = range(len(arm_q_seq) - 1, -1, -1) if reverse else range(len(arm_q_seq))
    direction = "reverse" if reverse else "forward"
    logger_mp.info(f"Replaying safe trajectory ({direction}, {len(arm_q_seq)} frames at {freq:.0f} Hz) ...")

    for i in indices:
        t0 = time.perf_counter()
        q = arm_q_seq[i]
        tau = arm_ik.solve_tau(q)
        arm_ctrl.ctrl_dual_arm(q, tau)

        if hand_seq is not None and ee_shared_mem is not None:
            h = hand_seq[i]
            try:
                from multiprocessing.sharedctypes import SynchronizedArray
                with ee_shared_mem["lock"]:
                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = h[:7].tolist()
                        ee_shared_mem["right"][:] = h[7:].tolist()
            except Exception:
                pass

        time.sleep(max(0, 1.0 / freq - (time.perf_counter() - t0)))

    logger_mp.info(f"Trajectory replay ({direction}) done.")


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
        HANG_HOME_Q = np.array([0.5, 0, 0, 1.2, -1.5708, -0.7, 0,
                                0.5, 0, 0, 1.2, 1.5708, -0.7, 0])

        safe_traj = None
        if cfg.safe_trajectory_path:
            safe_traj = np.load(cfg.safe_trajectory_path)
            logger_mp.info(f"Loaded safe trajectory from {cfg.safe_trajectory_path}")

        def go_home():
            logger_mp.info("Moving to home position (natural hang) ...")
            tau = arm_ik.solve_tau(HANG_HOME_Q)
            arm_ctrl.ctrl_dual_arm(HANG_HOME_Q, tau)
            time.sleep(2.0)

        def forward_to_init():
            """safe traj start -> forward replay -> dataset initial pose"""
            if safe_traj is not None:
                replay_trajectory(arm_ctrl, arm_ik, ee_shared_mem, safe_traj, reverse=False)
            logger_mp.info("Moving to dataset initial pose...")
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            if cfg.ee and left_ee_init is not None and right_ee_init is not None:
                try:
                    with ee_shared_mem["lock"]:
                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_init)
                            ee_shared_mem["right"][:] = to_list(right_ee_init)
                except Exception:
                    pass
            time.sleep(1.0)

        def reverse_to_start():
            """current pose -> safe traj end -> reverse replay -> safe traj start"""
            if safe_traj is not None:
                logger_mp.info("Moving back to trajectory end before reversing...")
                end_q = safe_traj["arm_q"][-1]
                tau = arm_ik.solve_tau(end_q)
                arm_ctrl.ctrl_dual_arm(end_q, tau)
                time.sleep(1.0)
                replay_trajectory(arm_ctrl, arm_ik, ee_shared_mem, safe_traj, reverse=True)

        def move_to_init_first_time():
            """home -> forward replay -> dataset initial pose (only at startup)"""
            go_home()
            forward_to_init()

        def reset_to_init():
            """current pose -> dataset initial pose directly"""
            logger_mp.info("Moving to dataset initial pose...")
            tau = arm_ik.solve_tau(init_arm_pose)
            arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
            if cfg.ee and left_ee_init is not None and right_ee_init is not None:
                try:
                    with ee_shared_mem["lock"]:
                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_init)
                            ee_shared_mem["right"][:] = to_list(right_ee_init)
                except Exception:
                    pass
            time.sleep(1.0)

        def safe_retreat():
            """current pose -> reverse replay -> home (for final exit)"""
            reverse_to_start()
            go_home()

        move_to_init_first_time()

        quit_program = False
        while not quit_program:
            user_input = input("Press 's' to start evaluation, 'r' to reset to initial pose, or 'q' to quit: ")
            user_input = user_input.strip().lower()
            print(f"user_input: {user_input}")

            if user_input == "q":
                logger_mp.info("Quit requested. Safe retreat...")
                safe_retreat()
                break

            if user_input == "r":
                logger_mp.info("Resetting to initial pose...")
                reset_to_init()
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
                full_state_np = np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0).astype(np.float32)
                state_tensor = torch.from_numpy(full_state_np).float()

                # Build observation in new GR00T 1.6 format: nested dict with (B, T, ...) numpy arrays
                observation["state"] = {
                    "left_arm": full_state_np[:7][np.newaxis, np.newaxis, :],
                    "right_arm": full_state_np[7:14][np.newaxis, np.newaxis, :],
                    "left_hand": full_state_np[14:21][np.newaxis, np.newaxis, :],
                    "right_hand": full_state_np[21:28][np.newaxis, np.newaxis, :],
                }
                task_str = step["task"] if step["task"] else ""
                observation["language"] = {
                    "annotation.human.task_description": [[task_str]],
                }

                # 2. Get Action from Policy
                action_dict, _ = policy.get_action(observation)
                actions = np.concatenate(
                    [action_dict[key] for key in action_dict.keys()],
                    axis=-1,
                )
                actions = actions[0]

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
                logger_mp.info("Quit during eval. Safe retreat...")
                safe_retreat()
                break

            # After a session ends with 'r', return to initial pose automatically
            logger_mp.info("Returning to initial pose...")
            reset_to_init()

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

    assert cfg.modality_config_path, "--modality_config_path is required (path to your modality config .py)"
    spec = importlib.util.spec_from_file_location("modality_config", cfg.modality_config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=cfg.model_path,
        device="cuda",
    )
    with torch.no_grad():
        eval_policy(policy, cfg=cfg, dataset=dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
