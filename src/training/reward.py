"""Hàm phần thưởng (reward) cho cờ tướng.

Thiết kế (đã sửa sau vòng huấn luyện 1e6 steps):

Vấn đề phát hiện được với reward kiểu "cộng dồn chênh lệch quân":

    r_t = material_coef * (material_do(t) - material_den(t))

Vì ``r_t`` phụ thuộc vào **trạng thái hiện tại** chứ không phải **thay đổi** do nước
đi tạo ra, tổng shaping của một ván ~100 nước lên tới **+12.5** trong khi phần
thưởng kết cục chỉ là **±1**. Hệ quả (đo thực tế trên checkpoint cũ):

    - 91/100 ván kéo dài hết 200 nước -> hoà;
    - agent học cách "ăn cho nhiều rồi giằng co", không học cách chiếu hết;
    - value head ``tanh`` bị kẹp trong [-1, 1] nên không thể biểu diễn return ~±12.

Cách sửa áp dụng ở đây — **potential-based shaping** (Ng et al., 1999):

    r_shaping(t) = gamma * Phi(s_{t+1}) - Phi(s_t),   Phi(s) = coef * (mat(color) - mat(opp))

Công thức này **bảo toàn chính sách tối ưu** (tổng shaping chỉ phụ thuộc trạng
thái cuối), nên chênh lệch quân chỉ còn là "đường gợi ý" mà không lấn át mục
tiêu thắng/thua. Thêm ``step_penalty`` để trừng phạt việc kéo dài ván.

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
    """Hệ số phần thưởng (đọc từ YAML).

    Thuộc tính:
        win              : phần thưởng kết cục thắng   (``+1.0``).
        loss             : phần thưởng kết cục thua    (``-1.0``).
        draw             : phần thưởng kết cục hoà     (``0 → -0.1`` để chống kéo giờ).
        material_coef    : hệ số potential Phi(s) = coef * (material_do - material_den).
        step_penalty     : phạt mỗi bước còn sống (chống kéo giờ).
        potential_gamma  : ``gamma`` dùng cho potential-based shaping (thường = 0.99).
    """

    win: float = 1.0
    loss: float = -1.0
    draw: float = -0.1
    material_coef: float = 0.01
    step_penalty: float = 0.005
    potential_gamma: float = 0.99

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "RewardConfig":
        """Khởi tạo từ mục ``reward`` trong file YAML (tương thích ngược)."""
        cfg = cfg or {}
        return cls(
            win=float(cfg.get("win", 1.0)),
            loss=float(cfg.get("loss", -1.0)),
            draw=float(cfg.get("draw", -0.1)),
            material_coef=float(cfg.get("material_coef", 0.01)),
            step_penalty=float(cfg.get("step_penalty", 0.005)),
            potential_gamma=float(cfg.get("potential_gamma", 0.99)),
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


def material_potential(board, color: int, coef: float) -> float:
    """Potential function ``Phi(s)`` cho potential-based shaping.

    ``Phi(s) = coef * (material(color) - material(đối thủ))`` — **một số vô hướng**,
    chỉ đo "bên nào đang lợi quân" chứ không phải phần thưởng.

    Args:
        board: ``cchess.ChessBoard``.
        color: Bên được tính potential.
        coef: Hệ số ``material_coef``.

    Returns:
        Giá trị Phi(s). Đặt ``coef == 0`` để tắt hoàn toàn shaping.
    """
    if coef == 0.0:
        return 0.0
    opponent = BLACK if color == RED else RED
    return coef * (material_value(board, color) - material_value(board, opponent))


def potential_shaping(
    phi_prev: float,
    phi_next: float,
    gamma: float = 0.99,
) -> float:
    """Potential-based shaping: ``gamma * Phi(s') - Phi(s)``.

    Khác với ``coef * Phi(s)``, công thức này **bảo toàn chính sách tối ưu**
    (Ng et al., 1999): tổng shaping của ván dài không còn tích luỹ nữa mà thu hẹp
    về trạng thái cuối. Từ đó phần thưởng thắng/thua (±1) là tín hiệu thống trị.

    Args:
        phi_prev: ``Phi(s)`` tính TRƯỚC nước đi (dùng :func:`material_potential`).
        phi_next: ``Phi(s')`` tính SAU nước đi.
        gamma: Hệ số cho term ``gamma * Phi(s')`` (thường = ``gamma`` của PPO).

    Returns:
        ``gamma * phi_next - phi_prev``.
    """
    return gamma * phi_next - phi_prev


# Giữ alias cũ để không phá vỡ import ngoài dự án.
def material_shaping(board, color: int, coef: float) -> float:
    """(BỊ THAY THẾ) Shaping cộng dồn kiểu cũ — **không dùng trong train nữa**.

    Hàm này được giữ lại chỉ để tương thích với code/test cũ. Shaping kiểu cũ
    cộng ``coef * Phi(s)`` mỗi bước nên tổng của một ván lên tới ~12.5 điểm —
    gấp 12 lần terminal reward — dẫn đến reward hacking (đo ở vòng train trước).
    Hãy dùng :func:`potential_shaping` cho mọi đường đi huấn luyện.
    """
    return material_potential(board, color, coef)


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
    phi_prev: float = 0.0,
) -> float:
    """Hàm phần thưởng tổng hợp — API chính của module.

    Args:
        board: ``cchess.ChessBoard`` (thế cờ HIỆN TẠI, sau nước đi).
        color: Bên vừa thực hiện nước đi (nhận phần thưởng).
        result: Kết cục nếu ván đã kết thúc (``"win"``/``"loss"``/``"draw"``).
        config: Hệ số phần thưởng; mặc định ``RewardConfig()``.
        terminal: Nếu ``True``, chỉ trả phần thưởng kết cục (không shaping).
        phi_prev: ``Phi(s)`` tín TRƯỚC nước đi (xem :func:`material_potential`).
            Bắt buộc phải tính **trước** khi gọi ``board.move()`` vì cchess sửa
            bàn cờ tại chỗ.

    Returns:
        Phần thưởng cho bên ``color``.

    Ghi chú:
        Với ``terminal=True`` chỉ trả ``±1``/``-0.1``. Ngược lại trả
        ``potential_shaping(phi_prev, Phi(s')) - step_penalty`` — trong đó
        ``step_penalty`` phạt mỗi bước để agent không kéo dài ván.
    """
    config = config or RewardConfig()

    if terminal:
        return terminal_reward(result, config)

    # Nước đi giữa ván: potential-based shaping + phạt bước.
    phi_next = material_potential(board, color, config.material_coef)
    shaping = potential_shaping(phi_prev, phi_next, gamma=config.potential_gamma)
    return shaping - config.step_penalty
