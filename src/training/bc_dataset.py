"""Dataset Behavioral Cloning cho cờ tướng từ parquet public (Pikafish pipeline).

Nguồn: ``ysong18/xiangqi-dataset`` (HuggingFace: 63k train + 65k test = ~128k
vị trí), mỗi dòng là một vị trí do pipeline Pikafish sinh ra:

    board90 : chuỗi 90 ký tự, hàng 0 = hàng trên; '.' = ô trống,
              chữ HOA = Đen, chữ thường = Đỏ (đúng quy ước FEN của cchess).
    side    : 'red' | 'black' — bên đang đi.
    played  : nước đã đi, ICCS ``file+rank`` (ví dụ 'h2e2').
    best    : nước tốt nhất theo engine.
    resultQ : điểm kết quả ván (dùng làm target cho value head).

Quy ước tọa độ — ĐÃ XÁC MINH thực nghiệm trên 2000 mẫu đầu:
với ``y_cchess == rank`` và ``x = file - 'a'``, 100.0% nước ``played`` là
hợp lệ theo ``cchess.ChessBoard``; nếu lật bàn (flip) chỉ còn 4.4%.
Vì vậy mapping dưới đây là đúng, không phải phỏng đoán.

Mỗi mẫu sinh ra: ``graph`` (PyG Data), ``legal_mask`` (8100,), ``action``
(from*90+to), ``value_target`` trong [-1,1] và ``weight`` ưu tiên nước ``best``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.env.constants import positions_to_action
from src.env.graph_builder import build_graph

logger = logging.getLogger(__name__)

DATASET_URLS = {
    "train": (
        "https://huggingface.co/api/datasets/ysong18/xiangqi-dataset"
        "/parquet/default/train/0.parquet"
    ),
    "test": (
        "https://huggingface.co/api/datasets/ysong18/xiangqi-dataset"
        "/parquet/default/test/0.parquet"
    ),
}
DATASET_URL = DATASET_URLS["train"]


def iccs_to_action(move: str) -> int:
    """ICCS ``'h2e2'`` -> action ``0..8099``.

    Args:
        move: Chuỗi 4 ký tự, ví dụ ``"h2e2"``.

    Returns:
        ``from_index * 90 + to_index``.
    """
    x1, y1 = ord(move[0]) - 97, int(move[1])
    x2, y2 = ord(move[2]) - 97, int(move[3])
    if not (0 <= x1 < 9 and 0 <= y1 < 10 and 0 <= x2 < 9 and 0 <= y2 < 10):
        raise ValueError(f"Nước ICCS ngoài bàn: {move}")
    return positions_to_action((x1, y1), (x2, y2))


def board90_to_grid(board90: str) -> List[List[Optional[str]]]:
    """Chuỗi 90 ký tự -> grid ``10 x 9`` (None = ô trống)."""
    if len(board90) != 90:
        raise ValueError(f"board90 phải dài 90, nhận {len(board90)}")
    grid: List[List[Optional[str]]] = []
    for y in range(10):
        row: List[Optional[str]] = []
        for x in range(9):
            ch = board90[y * 9 + x]
            row.append(None if ch == "." else ch)
        grid.append(row)
    return grid

class XiangqiBCDataSet(Dataset):
    """Dataset BC: ``(graph, legal_mask, action, value_target, weight)``.

    Args:
        parquet_path: File ``*.parquet`` ĐÃN tải — HOẶC danh sách nhiều file
            (train + test của ysong18 = ~128k vị trí, gấp đôi trước).
        label: ``"best"`` (nước tốt nhất theo engine — kiến nghị mới, chất
            lượng cao hơn ``"played"``) hoặc ``"played"``.
        max_samples: Giới hạn số mẫu (debug nhanh); ``None`` = toàn bộ.
        dedupe: Loại bỏ thế cờ trùng (board90+side) giữ lại bản ``best``
            đầu tiên (mặc định True — data engine có nhiều dòng trùng).
        min_abs_resultq: Chỉ giữ thế cờ có ``|resultQ| >= ngưỡng`` (hướng C —
            "data chuyên gia"). ``0.0`` = giữ tất cả.
        max_policy_kld: Chỉ giữ thế cờ mà engine **chắc chắn** về nước đi
            (``policyKld`` nhỏ = phân phối tập trung). ``None`` = giữ tất cả.
        require_best_move: Chỉ giữ dòng mà nước engine tốt nhất khớp với
            ``label`` (dữ liệu "thuần chuyên gia").

    BÀI HỌC từ 4 vòng train (17/10) — vì sao cần lọc:
        - pretrain 63k 'played' : val acc 0.094
        - pretrain 128k 'best'  : val acc 0.137  -> +46%
        - PPO warm trên 'best'  : Elo +55.8 -> +17.2 (TỆ), draw vẫn 86%
      => data đã tốt hơn nhưng PPO vẫn học hòa. Lọc theo CHẤT LƯỢNG ván
      (resultQ) và ĐỘ TIN CẬY của engine (policyKld) sẽ cho tập huấn luyện
      "sạch" hơn: chỉ thế cờ quyết định + nước đi engine chắc chắn.
    """

    def __init__(
        self,
        parquet_path: str | List[str],
        label: str = "best",
        max_samples: Optional[int] = None,
        dedupe: bool = True,
        min_abs_resultq: float = 0.0,
        max_policy_kld: Optional[float] = None,
        require_best_move: bool = False,
    ) -> None:
        import pyarrow.parquet as pq

        if label not in ("played", "best"):
            raise ValueError("label phải là 'played' hoặc 'best'")
        self.label = label

        paths = [parquet_path] if isinstance(parquet_path, str) else list(parquet_path)
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy dataset: {p}. Tải bằng:\n"
                    + "\n".join(
                        f"  curl -L -o {name}.parquet {url}"
                        for name, url in DATASET_URLS.items()
                    )
                )

        need_kld = max_policy_kld is not None
        need_best_flag = require_best_move
        read_cols = ["board", "side", "played", "best", "resultQ"]
        if need_kld:
            read_cols.append("policyKld")

        self.rows: List[Dict[str, Any]] = []
        n_skip = n_kld = n_resultq = n_best = 0
        seen = set()
        n_files = 0
        for p in paths:
            tbl = pq.read_table(str(p), columns=read_cols)
            cols = tbl.to_pydict()
            n = tbl.num_rows
            n_files += n
            for i in range(n):
                # C) Lọc theo độ tin cậy engine (KLD thấp = phân phối tập trung).
                if need_kld and cols["policyKld"][i] > max_policy_kld:
                    n_kld += 1
                    continue
                # C) Chỉ giữ thế cờ quyết định (ván có kết quả rõ ràng).
                if abs(float(cols["resultQ"][i] or 0.0)) < min_abs_resultq:
                    n_resultq += 1
                    continue
                # C) Chỉ giữ dòng mà nước 'played' khớp 'best' (chuyên gia).
                if need_best_flag and cols["played"][i] != cols["best"][i]:
                    n_best += 1
                    continue
                move = cols[label][i]
                if not move or len(move) != 4:
                    n_skip += 1
                    continue
                key = (cols["board"][i], cols["side"][i])
                if dedupe and key in seen:
                    continue
                try:
                    action = iccs_to_action(move)
                except ValueError:
                    n_skip += 1
                    continue
                seen.add(key)
                self.rows.append(
                    {
                        "board90": cols["board"][i],
                        "side": cols["side"][i],
                        "action": action,
                        "resultQ": float(cols["resultQ"][i] or 0.0),
                        "is_best": bool(cols["played"][i] == cols["best"][i]),
                    }
                )
                if max_samples is not None and len(self.rows) >= max_samples:
                    break
            if max_samples is not None and len(self.rows) >= max_samples:
                break
        if n_skip or n_kld or n_resultq or n_best:
            logger.info(
                "Lọc: lỗi=%d, KLD=%d, |resultQ|<%.3f:%d, !best=%d, giữ lại %d/%d.",
                n_skip, n_kld, min_abs_resultq, n_resultq, n_best,
                len(self.rows), n_files,
            )
        logger.info(
            "Nạp BC dataset: %d mẫu (dedupe %s) từ %d file (label=%s).",
            len(self.rows), "BẬT" if dedupe else "TẮT", len(paths), label,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from cchess import ChessBoard
        from cchess.common import BLACK, RED

        from src.env.xiangqi_env import XiangqiEnv

        row = self.rows[idx]
        grid = board90_to_grid(row["board90"])

        board = ChessBoard()
        board.clear()
        for y in range(10):
            for x in range(9):
                ch = grid[y][x]
                if ch is not None:
                    board._board[y][x] = ch
        # Dùng env chỉ để lấy legal mask (không reset về khai cuộc).
        env = XiangqiEnv(max_moves=200, legal_move_filter=False, seed=None)
        env.board = board
        # QUAN TRỌNG: ``cchess`` lưu lượt trong ``board.move_player`` và expose
        # qua ``set_move_color()``. Gán thẳng ``_next_turn`` sẽ khiến
        # ``get_move_color()`` trả 0 -> không sinh nước đi nào -> mask rỗng.
        board.set_move_color(RED if row["side"] == "red" else BLACK)
        env.move_count = 0
        env._mask_cache = None
        env._position_counts.clear()

        graph = build_graph(board)
        mask = env.get_legal_mask()

        action = row["action"]
        legal = bool(mask[action])
        value_target = max(-1.0, min(1.0, row["resultQ"]))
        weight = 2.0 if row["is_best"] else 1.0
        if not legal:
            weight = 0.0

        return {
            "graph": graph,
            "legal_mask": mask,
            "action": torch.tensor(action, dtype=torch.long),
            "value_target": torch.tensor(value_target, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def bc_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gộp list mẫu BC thành batch (PyG Batch + stack tensor)."""
    from torch_geometric.data import Batch

    return {
        "batch": Batch.from_data_list([b["graph"] for b in batch]),
        "legal_mask": torch.stack([b["legal_mask"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "value_target": torch.stack([b["value_target"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
    }