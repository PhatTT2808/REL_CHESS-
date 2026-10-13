"""Nạp cấu hình YAML, hợp nhất override và tiện ích liên quan.

Mọi đường dẫn trong dự án đều đọc từ config (mục ``paths``) — không hardcode.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Thư mục gốc của dự án = .../Project_REL (chứa configs/, scripts/, src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def load_config(
    config_path: str | os.PathLike[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Đọc file YAML cấu hình và hợp nhất với dict ``overrides``.

    Args:
        config_path: Đường dẫn tới file YAML (tương đối tính từ thư mục gốc dự án).
        overrides: Dict phẳng dạng ``{"training.lr": 1e-3}`` hoặc dict lồng nhau.
            Các key có dấu ``.`` sẽ được hiểu là đường dẫn lồng nhau.

    Returns:
        Dict cấu hình đã hợp nhất.

    Raises:
        FileNotFoundError: Nếu file cấu hình không tồn tại.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    config["_config_path"] = str(path)

    if overrides:
        config = _apply_overrides(config, overrides)

    _fill_defaults(config)
    return config


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Ghi đè cấu hình. Key chứa ``.`` được hiểu là đường dẫn lồng nhau."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            parts = key.split(".")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _fill_defaults(config: Dict[str, Any]) -> None:
    """Bổ sung các giá trị mặc định còn thiếu để tránh ``KeyError``."""
    config.setdefault("seed", 42)
    config.setdefault("paths", {})
    config["paths"].setdefault("checkpoints", "checkpoints")
    config["paths"].setdefault("logs", "logs")
    config["paths"].setdefault("figures", "figures")
    config["paths"].setdefault("pool", str(Path(config["paths"]["checkpoints"]) / "pool"))

    config.setdefault("env", {})
    config["env"].setdefault("max_moves", 200)
    config["env"].setdefault("perpetual_check_limit", 6)
    config["env"].setdefault("legal_move_filter", True)

    config.setdefault("training", {})
    config["training"].setdefault("device", "auto")
    config["training"].setdefault("num_workers", 0)
    config["training"].setdefault("use_amp", True)
    config["training"].setdefault("minibatch_size", 4096)
    config["training"].setdefault("pool_snapshot_interval", 5)
    config["training"].setdefault("max_moves", config["env"]["max_moves"])


def resolve_path(relative_or_absolute: str | os.PathLike[str]) -> Path:
    """Chuyển đường dẫn tương đối (trong config) thành tuyệt đối, và tạo thư mục cha.

    Args:
        relative_or_absolute: Đường dẫn, có thể tương đối so với thư mục gốc dự án.

    Returns:
        ``Path`` tuyệt đối.
    """
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa tồn tại và trả về ``Path``."""
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_device(device_cfg: str = "auto"):
    """Chọn thiết bị torch theo cấu hình.

    Args:
        device_cfg: ``"auto"`` (CUDA nếu có), hoặc ``"cpu"``, ``"cuda"``, ``"cuda:0"``...

    Returns:
        ``torch.device`` tương ứng.
    """
    import torch

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def resolve_num_workers(num_workers: int, device_cfg: str = "auto") -> int:
    """Suy ra số process sinh self-play song song.

    ``cchess`` là pure Python nên CPU là nút cổ chai thật; mặc định dùng
    toàn bộ nhân CPU trừ 1 (để chừa cho tiến trình chính và GPU transfer).
    """
    if num_workers and num_workers > 0:
        return num_workers
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)
