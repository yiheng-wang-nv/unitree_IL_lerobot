"""
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import json
import torch
import logging
import sys
import select
import queue
import threading

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
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

# Import TeleVuerWrapper for teleoperation intervention
try:
    from televuer import TeleVuerWrapper
except ImportError:
    print("Could not import TeleVuerWrapper. Make sure televuer is installed.")
    # We can continue without it, but teleop won't work. 
    # For now let's assume it's there or fail later if used.
    pass

# Import EpisodeWriter from teleop
try:
    from teleop.utils.episode_writer import EpisodeWriter
except ImportError:
    print("Could not import EpisodeWriter. Make sure rl/xr_teleoperate is in python path.")
    raise

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
        
        # Setup TeleVuer for intervention
        tv_wrapper = None
        try:
            tv_img_shm_name = image_info["shm_resources"][0].name
            tv_wrapper = TeleVuerWrapper(
                binocular=False,
                use_hand_tracking=False,
                img_shape=image_info["tv_img_shape"],
                img_shm_name=tv_img_shm_name,
                return_state_data=True,
                return_hand_rot_data=False,
                display_scale=0.1,
            )
            logger_mp.info("TeleVuerWrapper initialized for intervention.")
        except Exception as e:
            logger_mp.warning(f"Failed to initialize TeleVuerWrapper: {e}. Teleop intervention will not work.")

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

        episode_recorder: Optional[EpisodeWriter] = None
        if getattr(cfg, "record_data", False):
            if getattr(cfg, "record_dir", ""):
                record_dir = Path(cfg.record_dir).expanduser()
            else:
                base_root = Path(cfg.root).expanduser() if getattr(cfg, "root", "") else Path.cwd()
                record_dir = base_root / "recordings"
            record_dir = record_dir.resolve()
            logger_mp.info(f"Recording enabled. Saving episodes to {record_dir}.")
            
            # Initialize EpisodeWriter
            episode_recorder = EpisodeWriter(
                task_dir=str(record_dir),
                task_goal="evaluation",
                frequency=cfg.frequency,
                image_size=[tv_img_shape[1], tv_img_shape[0]], # width, height
                rerun_log=False
            )

        quit_program = False
        # Note: EpisodeWriter manages episode IDs internally based on directory content
        episode_counter = episode_recorder.episode_id if episode_recorder else 0
        
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
            episode_counter += 1
            logger_mp.info(
                f"Starting evaluation loop at {cfg.frequency} Hz (episode {episode_counter:04d}).\n"
                "Controls:\n"
                "  'r' + Enter: Stop & Reset episode\n"
                "  'q' + Enter: Quit program\n"
                "  'p' + Enter: Pause/Resume Policy\n"
                "     [While Paused]:\n"
                "       - Robot holds position until teleop is toggled\n"
                "       - Press 's' + Enter OR Left Controller 'A' to start/stop teleop control & recording\n"
                "       - After enabling teleop, squeeze the trigger once to \"arm\" the fingers (keeps current grip)\n"
                "       - Releasing the trigger opens the grasp relative to that armed pose\n"
            )
            idx = 0
            episode_stop_reason = "manual_stop"
            task_name = ""
            is_paused = False
            teleop_active = False  # Whether teleop control is currently active
            teleop_recording = False  # Recording flag (mirrors teleop_active)
            TELEOP_TRIGGER_ARM_THRESHOLD = 0.05
            def _set_teleop_mode(enable: bool, source: str) -> None:
                nonlocal teleop_active, teleop_recording
                if enable:
                    if not tv_wrapper:
                        logger_mp.warning("TeleVuerWrapper not ready, cannot start teleop control.")
                        return
                    teleop_active = True
                    teleop_recording = True
                    logger_mp.info(f"Teleop control & recording STARTED ({source}).")
                else:
                    if teleop_active or teleop_recording:
                        logger_mp.info(f"Teleop control & recording STOPPED ({source}).")
                    teleop_active = False
                    teleop_recording = False
            if hasattr(step, "get"):
                try:
                    task_name = step.get("task", "")
                except Exception:
                    task_name = ""
            elif isinstance(step, dict):
                task_name = step.get("task", "")

            if episode_recorder:
                if task_name:
                    episode_recorder.text['goal'] = task_name
                episode_recorder.create_episode()
                episode_recorder.set_label("unspecified")

            try:
                while True:
                    # Non-blocking command check (press key then Enter)
                    try:
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            cmd = sys.stdin.readline().strip().lower()
                            if cmd == "r":
                                logger_mp.info("Stop & reset requested.")
                                episode_stop_reason = "reset"
                                break
                            if cmd == "q":
                                logger_mp.info("Quit requested during evaluation loop.")
                                episode_stop_reason = "quit"
                                quit_program = True
                                break
                            if cmd == "p":
                                is_paused = not is_paused
                                if is_paused:
                                    logger_mp.info(
                                        "PAUSED (Policy Stopped). Press 's' or controller Left A to start teleop control & recording, or 'p' to resume policy."
                                    )
                                    _set_teleop_mode(False, "pause")
                                else:
                                    logger_mp.info("RESUMED (Policy Control).")
                                    _set_teleop_mode(False, "resume")
                                
                                if is_paused and not tv_wrapper:
                                    logger_mp.warning("TeleVuerWrapper not ready, cannot teleop. Robot will just hold position.")
                            if is_paused and cmd == "s":
                                _set_teleop_mode(not teleop_active, "keyboard 's'")
                    except Exception:
                        pass

                    # Check controller Left A button for recording toggle if in paused state
                    if is_paused and tv_wrapper:
                        try:
                            tele_data = tv_wrapper.get_motion_state_data()
                            # Left A button (mapped to 's' functionality)
                            la = bool(getattr(tele_data.tele_state, 'left_aButton', False))
                            
                            # Edge detection logic needs persistent state, but we don't have a global dict here.
                            # We can use a simple static attribute or just check current state with a debounce.
                            # For simplicity, let's implement a basic debounce using time.
                            current_time = time.time()
                            if not hasattr(eval_policy, "last_la_press_time"):
                                eval_policy.last_la_press_time = 0
                            
                            if la and (current_time - eval_policy.last_la_press_time > 0.5):
                                eval_policy.last_la_press_time = current_time
                                _set_teleop_mode(not teleop_active, "controller Left A")
                        except Exception:
                            pass

                    if is_paused:
                        loop_start_time = time.perf_counter()
                        
                        # --- Teleoperation Intervention Logic ---
                        if teleop_active and tv_wrapper:
                            # 1. Get Teleop Data
                            tele_data = tv_wrapper.get_motion_state_data()
                            
                            # 2. Get Robot State
                            current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                            
                            # 3. Solve IK
                            sol_q, sol_tauff = arm_ik.solve_ik(
                                tele_data.left_arm_pose, 
                                tele_data.right_arm_pose, 
                                current_lr_arm_q, 
                                current_lr_arm_dq
                            )
                            
                            # 4. Control Arm
                            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                            
                            # 5. Control Hands (Simple mapping for Dex3)
                            # Exactly as in teleop_dex3_controller.py - no extra logic
                            if cfg.ee == "dex3":
                                # Normalize triggers [10.0, 0.0] -> [0.0, 1.0]
                                left_trigger_raw = getattr(tele_data, 'left_trigger_value', 10.0)
                                right_trigger_raw = getattr(tele_data, 'right_trigger_value', 10.0)
                                left_trigger_norm = np.clip((10.0 - left_trigger_raw) / 10.0, 0.0, 1.0)
                                right_trigger_norm = np.clip((10.0 - right_trigger_raw) / 10.0, 0.0, 1.0)

                                # Constants from teleop script
                                THUMB1_MIN_RAD = 0.0
                                THUMB1_MAX_RAD = 55.0 * np.pi / 180.0
                                R_THUMB1_MIN_RAD = -40.0 * np.pi / 180.0
                                R_THUMB1_MAX_RAD = 0.0

                                with ee_shared_mem["lock"]:
                                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                                        l_cmd = np.array(ee_shared_mem["left"][:], dtype=float)
                                        r_cmd = np.array(ee_shared_mem["right"][:], dtype=float)

                                        # Direct mapping exactly as teleop_dex3_controller.py:
                                        # Left: trigger=0 (not pressed) -> THUMB1_MIN_RAD (closed)
                                        #       trigger=1 (pressed) -> THUMB1_MAX_RAD (open)
                                        l_cmd[1] = THUMB1_MIN_RAD + left_trigger_norm * (THUMB1_MAX_RAD - THUMB1_MIN_RAD)
                                        
                                        # Right: trigger=0 (not pressed) -> R_THUMB1_MAX_RAD (closed)
                                        #        trigger=1 (pressed) -> R_THUMB1_MIN_RAD (open)
                                        r_cmd[1] = R_THUMB1_MAX_RAD + right_trigger_norm * (R_THUMB1_MIN_RAD - R_THUMB1_MAX_RAD)

                                        ee_shared_mem["left"][:] = to_list(l_cmd)
                                        ee_shared_mem["right"][:] = to_list(r_cmd)
                        else:
                            # Fallback: just hold position if no teleop
                            current_q = arm_ctrl.get_current_dual_arm_q()
                            tau = arm_ik.solve_tau(current_q)
                            arm_ctrl.ctrl_dual_arm(current_q, tau)

                        # --- Recording Logic (Teleop) ---
                        if episode_recorder and teleop_recording:
                            # Capture current state for recording
                            current_tv_image = tv_img_array.copy()
                            current_wrist_image = wrist_img_array.copy()
                            
                            current_lr_arm_q_rec = arm_ctrl.get_current_dual_arm_q()
                            
                            left_ee_state_rec = np.array([])
                            right_ee_state_rec = np.array([])
                            if cfg.ee:
                                with ee_shared_mem["lock"]:
                                    full_state_rec = np.array(ee_shared_mem["state"][:])
                                    left_ee_state_rec = full_state_rec[:ee_dof]
                                    right_ee_state_rec = full_state_rec[ee_dof:]
                            
                            colors = {}
                            colors["color_0"] = current_tv_image
                            colors["color_1"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors["color_2"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                            
                            depths = {}
                            states = {
                                "left_arm": {"qpos": current_lr_arm_q_rec[:7].tolist(), "qvel": [], "torque": []},
                                "right_arm": {"qpos": current_lr_arm_q_rec[7:].tolist(), "qvel": [], "torque": []},
                                "left_ee": {"qpos": left_ee_state_rec.tolist() if len(left_ee_state_rec)>0 else [], "qvel": [], "torque": []},
                                "right_ee": {"qpos": right_ee_state_rec.tolist() if len(right_ee_state_rec)>0 else [], "qvel": [], "torque": []},
                                "body": {"qpos": []}
                            }
                            
                            # For actions in teleop mode, we use the current state/command as action
                            # Or if we solved IK, we use that as action. 
                            # Let's use the computed IK solution if available (sol_q), otherwise current q.
                            if tv_wrapper:
                                act_l_arm = sol_q[:7]
                                act_r_arm = sol_q[-7:]
                                # Hand actions? The command we just sent.
                                # We can just use the 'states' as 'actions' for simplicity or reconstruct it.
                                # To be consistent with policy loop, we should try to record the "commanded" action.
                                act_l_ee = []
                                act_r_ee = []
                                if cfg.ee == "dex3":
                                    # We reconstructed l_cmd/r_cmd above, but scope is tricky.
                                    # Let's read back what we wrote or just use current state approximation
                                    pass 
                            else:
                                act_l_arm = current_lr_arm_q_rec[:7]
                                act_r_arm = current_lr_arm_q_rec[7:]
                            
                            # Re-read hand commands for action recording
                            left_ee_act = []
                            right_ee_act = []
                            if cfg.ee:
                                with ee_shared_mem["lock"]:
                                    # Recording the INPUT command (left/right arrays) as action
                                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                                        left_ee_act = np.array(ee_shared_mem["left"][:]).tolist()
                                        right_ee_act = np.array(ee_shared_mem["right"][:]).tolist()

                            actions_rec = {
                                "left_arm": {"qpos": act_l_arm.tolist() if isinstance(act_l_arm, np.ndarray) else act_l_arm, "qvel": [], "torque": []},
                                "right_arm": {"qpos": act_r_arm.tolist() if isinstance(act_r_arm, np.ndarray) else act_r_arm, "qvel": [], "torque": []},
                                "left_ee": {"qpos": left_ee_act, "qvel": [], "torque": []},
                                "right_ee": {"qpos": right_ee_act, "qvel": [], "torque": []},
                                "body": {"qpos": []}
                            }
                            
                            episode_recorder.add_item(colors=colors, depths=depths, states=states, actions=actions_rec)

                        # Maintain frequency
                        time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                        continue

                    loop_start_time = time.perf_counter()

                    # 1. Get Observations
                    # This updates current_arm_q from robot
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

                        # CRITICAL: Capture images and state for recording NOW
                        current_tv_image = tv_img_array.copy()
                        current_wrist_image = wrist_img_array.copy()
                        
                        # Note: 'process_images_and_observations_gr00t' updates current_arm_q, so we should get fresh reading if we want exact sync with image
                        # But action execution loop might be faster than image update (30Hz vs control freq).
                        # process_images_and_observations_gr00t also does a sleep if called in loop? No, it just reads.
                        
                        if episode_recorder:
                            # Get current states
                            # Re-read arm state to be precise or use the one from start of loop?
                            # Teleop script reads arm state inside the loop.
                            current_lr_arm_q_rec = arm_ctrl.get_current_dual_arm_q()
                            
                            left_ee_state_rec = np.array([])
                            right_ee_state_rec = np.array([])
                            if cfg.ee:
                                with ee_shared_mem["lock"]:
                                    full_state_rec = np.array(ee_shared_mem["state"][:])
                                    left_ee_state_rec = full_state_rec[:ee_dof]
                                    right_ee_state_rec = full_state_rec[ee_dof:]
                                    
                            # Prepare data for EpisodeWriter
                            colors = {}
                            colors["color_0"] = current_tv_image
                            colors["color_1"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors["color_2"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                            
                            depths = {}
                            
                            states = {
                                "left_arm": {
                                    "qpos": current_lr_arm_q_rec[:7].tolist(),
                                    "qvel": [],
                                    "torque": []
                                },
                                "right_arm": {
                                    "qpos": current_lr_arm_q_rec[7:].tolist(),
                                    "qvel": [],
                                    "torque": []
                                },
                                "left_ee": {
                                    "qpos": left_ee_state_rec.tolist() if len(left_ee_state_rec)>0 else [],
                                    "qvel": [],
                                    "torque": []
                                },
                                "right_ee": {
                                    "qpos": right_ee_state_rec.tolist() if len(right_ee_state_rec)>0 else [],
                                    "qvel": [],
                                    "torque": []
                                },
                                "body": {
                                    "qpos": []
                                }
                            }
                            
                            # Parse actions
                            arm_action_rec = action_np[:arm_dof]
                            left_arm_act = arm_action_rec[:7]
                            right_arm_act = arm_action_rec[7:]
                            
                            left_ee_act = []
                            right_ee_act = []
                            if cfg.ee:
                                ee_action_start_idx = arm_dof
                                left_ee_act = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                                right_ee_act = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                            
                            actions_rec = {
                                "left_arm": {
                                    "qpos": left_arm_act.tolist(),
                                    "qvel": [],
                                    "torque": []
                                },
                                "right_arm": {
                                    "qpos": right_arm_act.tolist(),
                                    "qvel": [],
                                    "torque": []
                                },
                                "left_ee": {
                                    "qpos": left_ee_act.tolist() if len(left_ee_act)>0 else [],
                                    "qvel": [],
                                    "torque": []
                                },
                                "right_ee": {
                                    "qpos": right_ee_act.tolist() if len(right_ee_act)>0 else [],
                                    "qvel": [],
                                    "torque": []
                                },
                                "body": {
                                    "qpos": []
                                }
                            }
                            
                            episode_recorder.add_item(colors=colors, depths=depths, states=states, actions=actions_rec)

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
            except Exception:
                episode_stop_reason = "error"
                logger_mp.exception("Error during evaluation loop.")
                raise
            finally:
                if episode_recorder:
                    episode_recorder.save_episode()

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
        
        if episode_recorder:
            episode_recorder.close()


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

