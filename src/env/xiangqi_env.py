"""Gym-like wrapper cho ``cchess.ChessBoard``.

LƯU Ý QUAN TRỌNG VỀ API ``cchess`` (khác với python-chess):
    - Không có ``board.turn``      -> dùng ``board.get_move_color()``
    - Không có ``is_game_over()``  -> tự cài trong file này
    - Không có ``is_repetition()`` -> dùng FEN + đếm tần suất (zhash() bị bug)
    - Không có ``piece_map()``     -> dùng ``src.env.graph_builder.iter_pieces``
    - ``board.move()`` KHÔNG tự đổi lượt -> phải gọi ``board.next_turn()`` sau đó.

Không gian hành động: 8100 = 90 x 90, ``action = from_idx * 90 + to_idx``
với ``idx = y * 9 + x``.

QUAN TRỌNG VỀ ĐỘ HỢP LỆ:
    ``board.create_moves()`` CHỈ sinh nước theo luật đi của quân, **KHÔNG** lọc
    nước tự-chiếu-tướng (để lộ tướng mình). Trong khi ``no_moves()`` / ``is_checkmate()``
    LẠI có lọc. Nếu legal mask chỉ dựa trên ``create_moves()``, agent có thể chọn
    nước để lộ tướng -> ``no_moves()`` trả ``False`` -> không phát hiện hết cờ.
    Vì vậy mask được lọc thêm bằng ``is_checked_move()`` (tốn CPU nhưng đúng luật).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:  # pragma: no cover - phụ thuộc môi trường
    from cchess import ChessBoard
    from cchess.common import FULL_INIT_FEN
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Không import được thư viện 'cchess'. Hãy cài đặt: pip install cchess==1.25.5"
    ) from exc

from src.env.constants import (
    ACTION_SPACE,
    PIECE_VALUES,
    action_to_positions,
    positions_to_action,
)
from src.env.graph_builder import BLACK, RED, build_graph

logger = logging.getLogger(__name__)

# Kết quả ván cờ, theo góc nhìn "bên vừa thực hiện nước đi".
RESULT_NONE: int = 0
RESULT_WIN: int = 1
RESULT_LOSS: int = -1

# Nguyên nhân kết thúc (đưa vào ``info`` để thống kê).
REASON_ONGOING: str = "ongoing"
REASON_CHECKMATE: str = "checkmate"
REASON_STALEMATE: str = "stalemate"
REASON_REPETITION: str = "threefold_repetition"
REASON_PERPETUAL: str = "perpetual_check"
REASON_MAX_MOVES: str = "max_moves"
REASON_KING_CAPTURED: str = "king_captured"


@dataclass
class StepInfo:
    """Thông tin phụ trả về sau mỗi bước ``step``."""

    reason: str
    result: int
    move_count: int
    fen: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "result": self.result,
            "move_count": self.move_count,
            "fen": self.fen,
        }


class XiangqiEnv:
    """Môi trường cờ tướng theo giao diện Gym.

    Ví dụ:
        >>> env = XiangqiEnv(max_moves=200, seed=42)
        >>> state = env.reset()
        >>> mask = state["legal_mask"]         # BoolTensor(8100)
        >>> action = int(mask.nonzero()[0])    # chọn 1 nước hợp lệ
        >>> state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        max_moves: int = 200,
        perpetual_check_limit: int = 6,
        legal_move_filter: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Khởi tạo môi trường.

        Args:
            max_moves: Số nước tối đa; vượt ngưỡng -> hoà.
            perpetual_check_limit: Số lần liên tiếp bên đi đều chiếu tướng
                và lặp lại vị trí thì coi là "chiếu lặp" -> hoà.
            legal_move_filter: Lọc nước tự-chiếu-tướng khỏi legal mask
                (đúng luật nhưng tốn CPU hơn).
            seed: Seed cho việc lấy mẫu ngẫu nhiên nội bộ.
        """
        self.max_moves = int(max_moves)
        self.perpetual_check_limit = int(perpetual_check_limit)
        self.legal_move_filter = bool(legal_move_filter)
        self._rng = np.random.default_rng(seed)

        self.board: ChessBoard = ChessBoard(FULL_INIT_FEN)
        self.initial_fen: str = FULL_INIT_FEN
        self.move_count: int = 0
        self.history: List[str] = []
        self._position_counts: Counter = Counter()
        self._check_streak: int = 0
        self._last_player: int = 0
        self._mask_cache: Optional[Tuple[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Thuộc tính tiện ích
    # ------------------------------------------------------------------
    @property
    def current_player(self) -> int:
        """Bên đang tới lượt đi (``RED`` = 1 hoặc ``BLACK`` = 2)."""
        return int(self.board.get_move_color())

    @property
    def fen(self) -> str:
        """FEN hiện tại của thế cờ."""
        return self.board.to_fen()

    def _position_key(self) -> str:
        """Khoá định danh thế cờ để đếm lặp (3-fold repetition).

        KHÔNG dùng ``board.zhash()``: bản 1.25.5 có bug — hash không đổi sau
        một số nước đi (đã kiểm chứng: zhash giống nhau trước/sau nước ``N b9c7``),
        nên nếu dùng sẽ báo lặp sai. Thay vào đó dùng FEN nhưng **bỏ phần
        bên-đi-lượt** để chỉ so sánh thế cờ thực sự, kèm cờ màu xen kẽ bên đi
        để tránh gộp nhầm hai thế cờ khác lượt.
        """
        fen = self.board.to_fen()
        board_part = fen.split(" ")[0]
        side = fen.split(" ") if len(fen.split(" ")) > 1 else ["", "w"]
        mover = side[1] if len(side) > 1 else "w"
        return f"{board_part}|{mover}"


    # ------------------------------------------------------------------
    # API Gym
    # ------------------------------------------------------------------
    def reset(
        self, fen: Optional[str] = None, max_moves: Optional[int] = None
    ) -> Dict[str, Any]:
        """Đưa môi trường về thế cờ ban đầu.

        Args:
            fen: Thế cờ tuỳ chọn; mặc định dùng thế cờ khai cuộc chuẩn.
            max_moves: Ghi đè số nước tối đa cho ván này.

        Returns:
            Dict trạng thái ``{"fen", "graph", "legal_mask", "player"}``.
        """
        self.initial_fen = fen if fen else FULL_INIT_FEN
        self.board = ChessBoard(self.initial_fen)
        if max_moves is not None:
            self.max_moves = int(max_moves)

        self.move_count = 0
        self.history = []
        self._position_counts = Counter()
        self._check_streak = 0
        self._last_player = 0
        self._mask_cache = None
        self._position_counts[self._position_key()] += 1

        return self._build_state()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Thực hiện một nước đi.

        Reward trả về theo góc nhìn **bên vừa thực hiện nước đi**.

        Args:
            action: Số nguyên ``0..8099``.

        Returns:
            ``(state, reward, done, info)``.

        Raises:
            ValueError: Nếu hành động nằm ngoài legal mask.
        """
        from_pos, to_pos = action_to_positions(action)
        legal_mask = self.get_legal_mask()
        if not bool(legal_mask[action]):
            raise ValueError(
                f"Hành động không hợp lệ: {action} ({from_pos}->{to_pos}) "
                f"tại FEN={self.fen}"
            )

        prev_player = self.current_player
        move = self.board.move(from_pos, to_pos)
        if move is None:  # pragma: no cover - mask đã bảo đảm hợp lệ
            raise ValueError(f"cchess từ chối nước đi {from_pos}->{to_pos}")

        # QUAN TRỌNG: ``cchess.ChessBoard.move()`` KHÔNG tự đổi lượt trong bản
        # 1.25.5 (đã kiểm chứng: ``get_move_color()`` giữ nguyên sau khi đi).
        # Phải gọi ``next_turn()`` thủ công, nếu không cả hai bên sẽ cùng đi Đỏ.
        self.board.next_turn()

        self._last_player = prev_player
        self.move_count += 1
        self.history.append(move.to_iccs())
        self._mask_cache = None  # mask đã thay đổi sau khi đi

        self._position_counts[self._position_key()] += 1

        # Đếm chuỗi chiếu tướng liên tiếp kèm lặp thế (cho luật chiếu lặp).
        if self.board.is_checking():
            if self._position_counts[self._position_key()] > 1:
                # Bên vừa đi chiếu tướng VÀ thế cờ đã từng xuất hiện -> chuỗi chiếu lặp
                self._check_streak += 1
            else:
                self._check_streak = 0
        else:
            self._check_streak = 0

        done, reason = self._check_terminal(prev_player, move)
        reward = self._terminal_reward(reason)

        info = StepInfo(
            reason=reason,
            result=self._result_for(reason),
            move_count=self.move_count,
            fen=self.fen,
        ).as_dict()

        state = self._build_state()
        return state, reward, done, info

    def render(self, mode: str = "text") -> str:
        """Trả về biểu diễn văn bản của bàn cờ (không in ra stdout).

        Args:
            mode: Hiện chỉ hỗ trợ ``"text"``.

        Returns:
            Chuỗi nhiều dòng mô tả bàn cờ.
        """
        if mode != "text":
            raise ValueError(f"Chế độ render không hỗ trợ: {mode}")
        turn = "ĐỎ" if self.current_player == RED else "ĐEN"
        header = f"FEN: {self.fen} | lượt: {turn}"
        n_moves = int(self.get_legal_mask().sum().item())
        footer = f"Số nước hợp lệ: {n_moves} | đã đi: {self.move_count}/{self.max_moves}"
        return "\n".join([header, *self.board.text_view(), footer])

    # ------------------------------------------------------------------
    # Legal mask
    # ------------------------------------------------------------------
    def get_legal_mask(self) -> torch.Tensor:
        """Sinh mask ``BoolTensor(8100)`` cho các nước đi hợp lệ hiện tại.

        Mask được cache theo khoá thế cờ vì việc lọc ``is_checked_move`` khá tốn CPU.
        Trả về mask toàn ``False`` nếu ván đã kết thúc kiểu "mất tướng"
        (``cchess`` cho phép ăn tướng, khi đó ``create_moves()`` sẽ crash).
        """
        key = self._position_key()
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]

        mask = np.zeros(ACTION_SPACE, dtype=np.bool_)
        # Nếu một trong hai tướng đã biến mất -> ván kết thúc, không còn nước đi.
        if self._king_missing():
            tensor = torch.from_numpy(mask)
            self._mask_cache = (key, tensor)
            return tensor

        for from_pos, to_pos in self.board.create_moves():
            if self.legal_move_filter:
                try:
                    if self.board.is_checked_move(from_pos, to_pos):
                        # Nước đi để lộ tướng mình -> không hợp lệ
                        continue
                except Exception:  # pragma: no cover
                    continue
            mask[positions_to_action(from_pos, to_pos)] = True

        tensor = torch.from_numpy(mask)
        self._mask_cache = (key, tensor)
        return tensor

    def _king_missing(self) -> bool:
        """Kiểm tra có tướng nào đã bị ăn khỏi bàn (nên ván đã kết thúc)."""
        try:
            return (
                self.board.get_king(RED) is None or self.board.get_king(BLACK) is None
            )
        except Exception:  # pragma: no cover
            return True

    def legal_actions(self) -> List[int]:
        """Danh sách các hành động hợp lệ (chỉ số ``0..8099``)."""
        return self.get_legal_mask().nonzero(as_tuple=False).flatten().tolist()

    def action_to_iccs(self, action: int) -> str:
        """Chuyển hành động thành chuỗi ICCS (ví dụ ``"a0a1"``) để debug/log."""
        (x1, y1), (x2, y2) = action_to_positions(action)
        return f"{chr(ord('a') + x1)}{y1}{chr(ord('a') + x2)}{y2}"

    # ------------------------------------------------------------------
    # Trạng thái
    # ------------------------------------------------------------------
    def _build_state(self) -> Dict[str, Any]:
        """Tạo dict trạng thái trả về cho agent.

        An toàn cả khi ván đã kết thúc do mất tướng (``create_moves`` crash được
        chặn bởi ``get_legal_mask``; graph vẫn dựng bình thường).
        """
        fen = self.fen
        return {
            "fen": fen,
            "graph": build_graph(self.board, fen=fen),
            "legal_mask": self.get_legal_mask(),
            "player": self.current_player,
            "done_king_missing": self._king_missing(),
        }

    # ------------------------------------------------------------------
    # Luật kết thúc ván (tự cài vì cchess không có)
    # ------------------------------------------------------------------
    def _check_terminal(self, prev_player: int, move) -> Tuple[bool, str]:
        """Xác định ván đã kết thúc chưa và lý do.

        Thứ tự ưu tiên:
            1. Tướng bị ăn (an toàn cho mọi trường hợp).
            2. Chiếu bí / hết nước (bên sắp đi thua).
            3. Chiếu lặp liên tiếp.
            4. Lặp thế cờ 3 lần.
            5. Chạm trần số nước.
        """
        # 1. Ăn tướng
        captured = getattr(move, "captured", None)
        if captured and captured.lower() == "k":
            return True, REASON_KING_CAPTURED

        # 2. Chiếu bí / hết nước -> bên vừa đi thắng (stalemate cờ tướng = thua)
        if self.board.is_checkmate() or self.board.no_moves():
            return True, REASON_CHECKMATE

        # 3. Chiếu lặp liên tiếp
        if self._check_streak >= self.perpetual_check_limit:
            return True, REASON_PERPETUAL

        # 4. Lặp thế cờ 3 lần (dùng Zobrist hash)
        if self._position_counts[self._position_key()] >= 3:
            return True, REASON_REPETITION

        # 5. Trần số nước
        if self.move_count >= self.max_moves:
            return True, REASON_MAX_MOVES

        return False, REASON_ONGOING

    def _result_for(self, reason: str) -> int:
        """Kết quả theo góc nhìn **bên vừa đi**: +1 thắng, -1 thua, 0 hoà/đang chơi."""
        if reason in (REASON_CHECKMATE, REASON_KING_CAPTURED):
            return RESULT_WIN
        if reason in (REASON_REPETITION, REASON_PERPETUAL, REASON_MAX_MOVES):
            return RESULT_NONE  # hoà, quy ước 0
        return RESULT_NONE

    def _terminal_reward(self, reason: str) -> float:
        """Reward tức thời theo kết cục (sparse). Shaping material ở ``training.reward``."""
        result = self._result_for(reason)
        if result == RESULT_WIN:
            return 1.0
        if result == RESULT_LOSS:
            return -1.0
        return 0.0


def material_balance(board) -> float:
    """Tính chênh lệch quân (material) giữa ĐỎ và ĐEN theo ``PIECE_VALUES``.

    Args:
        board: ``cchess.ChessBoard``.

    Returns:
        ``(material_red - material_black)``.
    """
    from src.env.graph_builder import iter_pieces

    total = 0.0
    for _x, _y, fench, color in iter_pieces(board):
        value = PIECE_VALUES.get(fench.lower(), 0.0)
        total += value if color == RED else -value
    return total

