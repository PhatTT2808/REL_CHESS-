"""Lưu / nạp checkpoint (trọng số, optimizer, lịch sử huấn luyện).

Mọi đường dẫn được truyền vào từ config (``paths.checkpoints``) — không hardcode.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.config import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    checkpoint_dir: str,
    extra: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> Path:
    """Lưu checkpoint ra ``checkpoint_dir/step_<step>.pt``.

    Args:
        model: Mô hình cần lưu (chỉ lưu ``state_dict``).
        optimizer: Optimizer (nếu có).
        step: Bước huấn luyện hiện tại.
        checkpoint_dir: Thư mục lưu checkpoint (từ config).
        extra: Dict thông tin bổ sung (epoch, metrics...).
        filename: Tên file tuỳ chọn; mặc định ``step_<step>.pt``.

    Returns:
        Đường dẫn file checkpoint đã lưu.
    """
    directory = ensure_dir(checkpoint_dir)
    name = filename or f"step_{step}.pt"
    path = directory / name

    payload: Dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    logger.info("Đã lưu checkpoint: %s", path)
    return path


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Nạp checkpoint vào mô hình (và optimizer nếu cung cấp).

    Args:
        path: Đường dẫn file checkpoint.
        model: Mô hình đích.
        optimizer: Optimizer đích (tuỳ chọn).
        map_location: Thiết bị để map tensor khi nạp.

    Returns:
        Dict metadata (``step``, ``extra``...).

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {resolved}")

    payload = torch.load(resolved, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    logger.info("Đã nạp checkpoint: %s (step=%s)", resolved, payload.get("step"))
    return {"step": payload.get("step", 0), "extra": payload.get("extra", {})}


def list_checkpoints(checkpoint_dir: str) -> list[tuple[int, Path]]:
    """Liệt kê checkpoint trong thư mục, sắp xếp theo số bước tăng dần.

    Args:
        checkpoint_dir: Thư mục chứa checkpoint.

    Returns:
        Danh sách ``[(step, path), ...]``.
    """
    directory = resolve_path(checkpoint_dir)
    if not directory.is_dir():
        return []

    results: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        match = CHECKPOINT_PATTERN.match(entry.name)
        if match:
            results.append((int(match.group(1)), entry))
    results.sort(key=lambda item: item[0])
    return results


def latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Trả về checkpoint mới nhất trong thư mục, hoặc ``None`` nếu chưa có."""
    items = list_checkpoints(checkpoint_dir)
    return items[-1][1] if items else None


def save_metrics(metrics: Dict[str, Any], metrics_path: str) -> Path:
    """Ghi thêm một bản ghi metrics vào file JSONL (mỗi dòng 1 JSON).

    Args:
        metrics: Dict metrics của một lần log.
        metrics_path: Đường dẫn file JSONL.

    Returns:
        Đường dẫn file đã ghi.
    """
    path = resolve_path(metrics_path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return path


def load_metrics(metrics_path: str) -> list[Dict[str, Any]]:
    """Đọc toàn bộ bản ghi metrics từ file JSONL (dùng cho visualize)."""
    path = resolve_path(metrics_path)
    if not path.is_file():
        return []
    records: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
