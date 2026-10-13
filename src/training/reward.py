"""Hàm phần thưởng (reward) cố định cho cờ tướng.

Thiết kế:
    - **Sparse terminal reward**: +1 thắng, -1 thua, 0 hoà.
    - **Material shaping nhẹ** (tuỳ chọn): cộng ``material_coef`` x chênh lệch
      quân sau mỗi nước, giúp agent học nhanh hơn ở giai đoạn đầu khi tín hiệu
      thắng/thua quá thưa. Đặt ``material_coef: 0`` để tắt hoàn toàn.

Toàn bộ hệ số đọc từ config — không hardcode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.env.constants import PIECE_VALUES
from src.env.graph_builder import BLACK, RED, iter_pieces

# Kết quả ván cờ theo góc nhìn người chơi.
RESULT_WIN: str = "win"
RESULT_LOSS: str = "loss"
RESULT_DRAW: str = "draw"


@dataclass
class RewardConfig:
    """Hệ số phần thưởng."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    material_coef: float = 0.01

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "RewardConfig":
        """Khởi tạo từ mục ``reward`` trong file YAML."""
        cfg = cfg or {}
        return cls(
            win=float(cfg.get("win", 1.0)),
            loss=float(cfg.get("loss", -1.0)),
            draw=float(cfg.get("draw", 0.0)),
            material_coef=float(cfg.get("material_coef", 0.01)),
        )


def material_value(board, color: int) -> float:
    """Tổng giá trị quân của một bên.

    Args:
        board: ``cchess.ChessBoard``.
        color: ``RED`` (1) hoặc ``BLACK`` (2).

    Returns:
        Tổng giá trị quân của ``color``.
    """
    total = 0.0
    for _x, _y, fench, piece_color in iter_pieces(board):
        if piece_color == color:
            total += PIECE_VALUES.get(fench.lower(), 0.0)
    return total


def material_shaping(board, color: int, coef: float) -> float:
    """Phần thưởng shaping theo chênh lệch quân, tính cho ``color``.

    Args:
        board: ``cchess.ChessBoard``.
        color: Bên được tính phần thưởng.
        coef: Hệ số nhân (``material_coef``).

    Returns:
        ``coef * (material(color) - material(đối thủ))``.
    """
    if coef == 0.0:
        return 0.0
    opponent = BLACK if color == RED else RED
    return coef * (material_value(board, color) - material_value(board, opponent))


def terminal_reward(result: Optional[str], config: RewardConfig) -> float:
    """Phần thưởng theo kết cục ván.

    Args:
        result: ``"win"`` / ``"loss"`` / ``"draw"`` / ``None`` (chưa kết thúc).
        config: Hệ số phần thưởng.

    Returns:
        Giá trị phần thưởng kết cục.
    """
    if result == RESULT_WIN:
        return config.win
    if result == RESULT_LOSS:
        return config.loss
    if result == RESULT_DRAW:
        return config.draw
    return 0.0


def compute_reward(
    board,
    color: int,
    result: Optional[str] = None,
    config: Optional[RewardConfig] = None,
    terminal: bool = False,
) -> float:
    """Hàm phần thưởng tổng hợp — dùng làm API chính của module.

    Args:
        board: ``cchess.ChessBoard`` (thế cờ HIỆN TẠI).
        color: Bên vừa thực hiện nước đi (nhận phần thưởng).
        result: Kết cục nếu ván đã kết thúc (``"win"``/``"loss"``/``"draw"``).
        config: Hệ số phần thưởng; mặc định ``RewardConfig()``.
        terminal: Nếu ``True``, chỉ trả phần thưởng kết cục (không shaping).

    Returns:
        Phần thưởng cho bên ``color``.

    Ví dụ:
        >>> cfg = RewardConfig(win=1.0, loss=-1.0, draw=0.0, material_coef=0.01)
        >>> compute_reward(board, color=RED, result="win", config=cfg, terminal=True)
        1.0
    """
    config = config or RewardConfig()

    if terminal:
        return terminal_reward(result, config)

    # Nước đi giữa ván: chỉ có shaping material (nếu bật).
    return material_shaping(board, color, config.material_coef)
