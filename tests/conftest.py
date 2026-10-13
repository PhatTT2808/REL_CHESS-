"""Cấu hình chung cho pytest: fixture dùng chung và sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Đảm bảo import được ``src.*`` khi chạy pytest từ thư mục gốc dự án.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _seed_everything() -> None:
    """Đặt seed cố định cho toàn bộ phiên test (reproducible)."""
    set_seed(42)


@pytest.fixture(scope="session")
def smoke_config() -> dict:
    """Nạp cấu hình smoke (nhỏ, chạy nhanh)."""
    return load_config("configs/smoke.yaml")


@pytest.fixture()
def env():
    """Một ``XiangqiEnv`` mới ở thế cờ khai cuộc."""
    from src.env import XiangqiEnv

    return XiangqiEnv(max_moves=200, seed=42)


@pytest.fixture()
def small_model():
    """Model nhỏ (nhanh) để test forward/backward."""
    from src.models import XiangqiPolicyNet

    return XiangqiPolicyNet(
        gnn_hidden_dim=32,
        gnn_layers=3,
        gnn_heads=4,
        motif_dim=16,
        num_motifs=8,
        policy_hidden=64,
        value_hidden=32,
        dropout=0.0,
    )


# Một FEN có sẵn thế "đỏ chiếu bí đen" để test trạng thái kết thúc.
CHECKMATE_FEN = "3k5/9/9/9/9/3R5/9/9/9/4K4 b w"
