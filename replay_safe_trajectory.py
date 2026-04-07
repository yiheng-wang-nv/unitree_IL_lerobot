"""
Replay a recorded safe trajectory (.npz) on the robot using unitree_IL_lerobot's IK.

Usage:
    python replay_safe_trajectory.py \
        --file safe_traj.npz \
        --arm G1_29 --ee dex3 \
        [--speed 1.0] [--sim]

Controls:
    Press 's' then Enter to start forward replay.
    After forward replay, press Enter to do reverse replay.
    Ctrl+C to abort at any time.
"""

import time
import argparse
import numpy as np

from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
from unitree_lerobot.eval_robot.utils.utils import to_list, EvalRealConfig

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def replay(arm_ctrl, arm_ik, ee_shared_mem, traj_data, reverse=False, speed=1.0):
    arm_q_seq = traj_data["arm_q"]
    hand_seq = traj_data["hand_state"] if "hand_state" in traj_data else None
    freq = float(traj_data["frequency"]) * speed

    indices = range(len(arm_q_seq) - 1, -1, -1) if reverse else range(len(arm_q_seq))
    direction = "REVERSE" if reverse else "FORWARD"
    logger_mp.info(f"Replaying {direction} ({len(arm_q_seq)} frames at {freq:.0f} Hz) ...")

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

    logger_mp.info(f"Replay {direction} done.")


def main():
    parser = argparse.ArgumentParser(description="Replay a safe trajectory .npz file")
    parser.add_argument("--file", type=str, required=True, help="Path to .npz trajectory file")
    parser.add_argument("--arm", type=str, default="G1_29")
    parser.add_argument("--ee", type=str, default="dex3")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    traj = np.load(args.file)
    arm_q = traj["arm_q"]
    freq = float(traj["frequency"])
    n = len(arm_q)
    logger_mp.info(f"Loaded {args.file}: {n} frames, {freq:.0f} Hz, {n/freq:.2f} s")
    logger_mp.info(f"Start q: {np.array2string(arm_q[0], precision=3)}")
    logger_mp.info(f"End   q: {np.array2string(arm_q[-1], precision=3)}")

    cfg = EvalRealConfig(
        repo_id="dummy",
        arm=args.arm,
        ee=args.ee,
        sim=args.sim,
        frequency=freq,
    )
    robot_interface = setup_robot_interface(cfg)
    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    ee_shared_mem = robot_interface["ee_shared_mem"]

    try:
        HANG_HOME_Q = np.array([0.5, 0, 0, 1.2, -1.5708, -0.7, 0,
                                0.5, 0, 0, 1.2, 1.5708, -0.7, 0])
        logger_mp.info("Moving to home position (natural hang) ...")
        tau = arm_ik.solve_tau(HANG_HOME_Q)
        arm_ctrl.ctrl_dual_arm(HANG_HOME_Q, tau)
        time.sleep(2.0)
        logger_mp.info("At home position.")

        user_input = input("Press 's' to start FORWARD replay: ")
        if user_input.strip().lower() != "s":
            logger_mp.info("Aborted.")
            return

        replay(arm_ctrl, arm_ik, ee_shared_mem, traj, reverse=False, speed=args.speed)

        input("\nForward done. Press Enter to replay REVERSE (Ctrl+C to abort)...")
        replay(arm_ctrl, arm_ik, ee_shared_mem, traj, reverse=True, speed=args.speed)

        logger_mp.info("All done. Robot should be back at start position.")
    except KeyboardInterrupt:
        logger_mp.info("Aborted.")


if __name__ == "__main__":
    main()
