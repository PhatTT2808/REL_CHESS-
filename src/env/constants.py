"""Hằng số dùng chung: mã hóa quân cờ, không gian hành động, giá trị quân.

Toàn bộ dự án dùng thống nhất các hằng số ở đây để tránh lệch logic giữa
env / graph builder / reward / model.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------
# Kích thước bàn cờ (cờ tướng: 9 cột x 10 hàng)
# ---------------------------------------------------------------------
BOARD_WIDTH: int = 9   # trục x (cột), 0..8
BOARD_HEIGHT: int = 10  # trục y (hàng), 0..9
NUM_SQUARES: int = BOARD_WIDTH * BOARD_HEIGHT  # 90
ACTION_SPACE: int = NUM_SQUARES * NUM_SQUARES  # 8100 = from * 90 + to

# ---------------------------------------------------------------------
# Ánh xạ quân cờ (FEN char <-> index) — 7 loại quân
#   K/k: Tướng, A/a: Sĩ, B/b: Tượng, N/n: Mã, R/r: Xe, C/c: Pháo, P/p: Binh/Tốt
# ---------------------------------------------------------------------
PIECE_TYPES: Tuple[str, ...] = ("k", "a", "b", "n", "r", "c", "p")
PIECE_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(PIECE_TYPES)}
NUM_PIECE_TYPES: int = len(PIECE_TYPES)

# Số chiều feature của 1 node: [x_norm, y_norm, onehot(7), is_red] = 10
NODE_FEATURE_DIM: int = 2 + NUM_PIECE_TYPES + 1

# ---------------------------------------------------------------------
# Giá trị quân cờ (dùng cho reward shaping "material").
# Tướng = 0 vì ăn tướng đã là trạng thái kết thúc (không cần shaping).
# ---------------------------------------------------------------------
PIECE_VALUES: Dict[str, float] = {
    "k": 0.0,
    "a": 2.0,
    "b": 2.0,
    "n": 4.0,
    "r": 9.0,
    "c": 4.5,
    "p": 1.0,
}

# ---------------------------------------------------------------------
# Loại cạnh trong đồ thị (edge type) — dùng làm edge_attr cho GATConv
# ---------------------------------------------------------------------
EDGE_ATTACK: int = 0   # A có thể ăn B (khác màu)
EDGE_DEFEND: int = 1   # A bảo vệ B (cùng màu)
EDGE_SAMELINE: int = 2  # A và B cùng hàng hoặc cùng cột
NUM_EDGE_TYPES: int = 3


def pos_to_index(x: int, y: int) -> int:
    """Chuyển toạ độ ``(x, y)`` (x: cột 0..8, y: hàng 0..9) thành chỉ số ``0..89``."""
    return y * BOARD_WIDTH + x


def index_to_pos(index: int) -> Tuple[int, int]:
    """Chuyển chỉ số ô ``0..89`` thành toạ độ ``(x, y)``."""
    return index % BOARD_WIDTH, index // BOARD_WIDTH


def action_to_positions(action: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Giải mã hành động ``0..8099`` thành cặp toạ độ ``(from_pos, to_pos)``.

    Quy ước: ``action = from_index * 90 + to_index``.
    """
    if not 0 <= action < ACTION_SPACE:
        raise ValueError(f"Hành động ngoài phạm vi [0, {ACTION_SPACE}): {action}")
    from_index, to_index = divmod(action, NUM_SQUARES)
    return index_to_pos(from_index), index_to_pos(to_index)


def positions_to_action(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """Mã hóa cặp toạ độ ``(from_pos, to_pos)`` thành hành động ``0..8099``."""
    return pos_to_index(*from_pos) * NUM_SQUARES + pos_to_index(*to_pos)
