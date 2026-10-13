"""Tiện ích đặt seed để đảm bảo kết quả tái lập (reproducible)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Đặt seed cho ``random``, ``numpy`` và ``torch``.

    Args:
        seed: Giá trị seed dùng chung.
        deterministic: Nếu ``True``, bật chế độ deterministic của cuDNN
            (chậm hơn nhưng kết quả ổn định hơn).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch luôn có trong dự án này
        pass


def seed_worker(worker_id: int) -> None:  # pragma: no cover - dùng cho DataLoader
    """Seed cho từng worker của DataLoader (tránh trùng dữ liệu giữa worker)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
