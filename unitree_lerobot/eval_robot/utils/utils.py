import numpy as np
import torch
import argparse
from typing import Any, Dict
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            #     if "images" in name:
            #         observation[name] = observation[name].type(torch.float32) / 255
            #         observation[name] = observation[name].permute(2, 0, 1).contiguous()

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


def cleanup_resources(image_info: Dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on the robot.")
    parser.add_argument("--repo_id", type=str, help="lerobot repo_id")
    parser.add_argument("--root", type=str, default="", help="root directory for the dataset")
    parser.add_argument("--episodes", type=int, default=0, help="episodes to evaluate")
    parser.add_argument("--frequency", type=float, default=30.0, help="data's frequency")

    # Basic control parameters
    parser.add_argument(
        "--arm", type=str, choices=["G1_29", "G1_23", "H1_2", "H1"], default="G1_29", help="Select arm controller"
    )
    parser.add_argument(
        "--ee",
        type=str,
        choices=["dex1", "dex3", "inspire1", "brainco"],
        default="dex3",
        help="Select end effector controller",
    )

    # Mode flags
    parser.add_argument("--motion", action="store_true", help="Enable motion control mode")
    parser.add_argument("--headless", action="store_true", help="Enable headless mode (no display)")
    parser.add_argument("--sim", action="store_true", help="Enable isaac simulation mode")
    parser.add_argument("--visualization", action="store_true", help="Rerun visualization")
    parser.add_argument("--send_real_robot", action="store_true", help="Enable execute on real robot mode")

    return parser.parse_args()


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    sim: bool = False
    visualization: bool = False
    send_real_robot: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
