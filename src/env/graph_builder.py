"""Chuyển thế cờ (ChessBoard) thành đồ thị PyG ``Data``.

Ý tưởng Neural-Symbolic: thay vì đưa ảnh bàn cờ vào CNN, ta biểu diễn thế cờ
như một **đồ thị quan hệ** giữa các quân cờ, để GNN học được cấu trúc
tấn công / phòng thủ / cùng đường.

Mỗi quân cờ = 1 node với feature:
    [x_norm, y_norm, onehot_loai_quan(7), is_red(1)]  -> 10 chiều

Các loại cạnh (edge_attr, 3 loại):
    0 = attack  : A có thể ăn B (khác màu, thoả luật đi của A, kể cả pháo nhảy)
    1 = defend  : A và B cùng màu (A bảo vệ B)
    2 = sameline: A và B cùng hàng hoặc cùng cột
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.env.constants import (
    EDGE_ATTACK,
    EDGE_DEFEND,
    EDGE_SAMELINE,
    NODE_FEATURE_DIM,
    NUM_EDGE_TYPES,
    PIECE_TO_IDX,
    PIECE_TYPES,
)

logger = logging.getLogger(__name__)

# Màu: cchess quy ước RED=1, BLACK=2 (xem cchess.common)
RED = 1
BLACK = 2


def _piece_species(fench: str) -> str:
    """Lấy ký tự loại quân (chữ thường) từ ký tự FEN, ví dụ ``"R"`` -> ``"r"``."""
    return fench.lower()


def iter_pieces(board) -> List[Tuple[int, int, str, int]]:
    """Duyệt toàn bộ quân cờ trên bàn.

    Args:
        board: ``cchess.ChessBoard``.

    Returns:
        Danh sách ``(x, y, fench, color)`` cho từng quân còn trên bàn.
        ``color`` là ``RED`` (1) hoặc ``BLACK`` (2).
    """
    pieces: List[Tuple[int, int, str, int]] = []
    grid = board._board  # list[10][9], None hoặc ký tự FEN
    for y in range(10):
        row = grid[y]
        for x in range(9):
            fench = row[x]
            if not fench:
                continue
            color = RED if fench.isupper() else BLACK
            pieces.append((x, y, fench, color))
    return pieces


def build_node_features(pieces: List[Tuple[int, int, str, int]]) -> np.ndarray:
    """Tạo ma trận feature node ``(N, 10)``.

    Feature gồm: toạ độ chuẩn hoá ``[0,1]``, one-hot loại quân (7), cờ màu đỏ (1).
    """
    features = np.zeros((len(pieces), NODE_FEATURE_DIM), dtype=np.float32)
    for i, (x, y, fench, color) in enumerate(pieces):
        species = _piece_species(fench)
        features[i, 0] = x / 8.0
        features[i, 1] = y / 9.0
        features[i, 2 + PIECE_TO_IDX[species]] = 1.0
        features[i, 2 + len(PIECE_TYPES)] = 1.0 if color == RED else 0.0
    return features


def _can_attack(board, fench: str, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
    """Kiểm tra quân ``fench`` ở ``from_pos`` có thể ăn quân ở ``to_pos`` không.

    Dùng chính luật đi của cchess (đã bao gồm chặn đường, pháo nhảy, tốt qua sông...).
    """
    target = board.get_fench(to_pos)
    if not target:
        return False
    target_is_upper = target.isupper()
    from_is_upper = fench.isupper()
    if target_is_upper == from_is_upper:  # cùng màu -> không phải tấn công
        return False
    try:
        piece = board.get_piece(from_pos)
        if piece is None:
            return False
        return bool(piece.is_valid_move(to_pos))
    except Exception:  # pragma: no cover - phòng thủ với input bất thường
        return False


def build_graph(board, fen: Optional[str] = None) -> Data:
    """Chuyển ``ChessBoard`` thành ``torch_geometric.data.Data``.

    Args:
        board: ``cchess.ChessBoard`` đã nạp thế cờ.
        fen: Nếu cung cấp, lưu kèm vào ``Data`` để truy vết.

    Returns:
        ``Data`` với:
        - ``x``: ``(N, 10)`` float32
        - ``edge_index``: ``(2, E)`` int64
        - ``edge_attr``: ``(E, 3)`` float32 (one-hot loại cạnh)
        - ``node_pos``: ``(N, 2)`` int64 — toạ độ ``(x, y)`` của từng node
        - ``fen``: chuỗi FEN (nếu có)
    """
    pieces = iter_pieces(board)
    node_pos = np.array([[x, y] for x, y, _, _ in pieces], dtype=np.int64)
    x = build_node_features(pieces)

    num_nodes = len(pieces)
    src: List[int] = []
    dst: List[int] = []
    types: List[int] = []

    # Xây dựng cạnh: O(N^2) nhưng N <= 32 nên rất nhanh.
    for i in range(num_nodes):
        xi, yi, fi, ci = pieces[i]
        for j in range(num_nodes):
            if i == j:
                continue
            xj, yj, fj, cj = pieces[j]
            if ci == cj:
                # Cùng màu -> cạnh phòng thủ
                src.append(i)
                dst.append(j)
                types.append(EDGE_DEFEND)
            elif _can_attack(board, fi, (xi, yi), (xj, yj)):
                # Khác màu và i ăn được j -> cạnh tấn công
                src.append(i)
                dst.append(j)
                types.append(EDGE_ATTACK)
            elif xi == xj or yi == yj:
                # Cùng hàng/cột -> cạnh quan hệ đường
                src.append(i)
                dst.append(j)
                types.append(EDGE_SAMELINE)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(types, dtype=torch.long)
        edge_attr = torch.nn.functional.one_hot(
            edge_type, num_classes=NUM_EDGE_TYPES
        ).float()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float32)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.node_pos = torch.from_numpy(node_pos)
    data.num_nodes = num_nodes
    if fen is not None:
        data.fen = fen
    return data


def empty_graph(fen: str = "") -> Data:
    """Đồ thị rỗng (dùng khi bàn không còn quân) — giữ shape hợp lệ cho model."""
    data = Data(
        x=torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float32),
    )
    data.node_pos = torch.zeros((0, 2), dtype=torch.long)
    data.num_nodes = 0
    data.fen = fen
    return data
