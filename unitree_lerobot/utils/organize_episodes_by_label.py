#!/usr/bin/env python3
"""Group Unitree teleop episodes into success/failure/unspecified folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

LABELS = ("success", "failure", "unspecified")


def _read_label(episode_dir: Path) -> str:
    label_path = episode_dir / "label.json"
    if not label_path.exists():
        return "unspecified"
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception:
        return "unspecified"
    raw = str(data.get("label", "unspecified")).strip().lower()
    if raw not in LABELS:
        return "unspecified"
    return raw


def _ensure_label_dirs(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for label in LABELS:
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        mapping[label] = label_dir
    return mapping


def _iter_episode_dirs(parent: Path) -> Iterable[Path]:
    for child in parent.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            yield child


def _move_episode(episode_dir: Path, target_dir: Path) -> None:
    destination = target_dir / episode_dir.name
    if destination.exists():
        raise FileExistsError(f"Destination {destination} already exists.")
    shutil.move(str(episode_dir), destination)
    print(f"Moved {episode_dir} -> {destination}")


def organise(root: Path) -> None:
    label_dirs = _ensure_label_dirs(root)

    def process_container(container: Path) -> None:
        for episode in list(_iter_episode_dirs(container)):
            label = _read_label(episode)
            target = label_dirs[label]
            if episode.parent == target:
                continue
            _move_episode(episode, target)

    process_container(root)

    for label_dir in label_dirs.values():
        process_container(label_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group episode directories by label.")
    parser.add_argument("root", type=Path, help="Directory containing episode_XXXX folders.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist.")
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")

    organise(root)


if __name__ == "__main__":
    main()
