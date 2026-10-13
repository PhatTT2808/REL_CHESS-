"""Đấu ván giữa các agent cờ tướng (Model, Random, Engine UCI/UCCI).

Cung cấp:
    - ``Agent`` (interface), ``RandomAgent``, ``ModelAgent``, ``EngineAgent``.
    - ``play_match``: chạy N ván xen kẽ Đỏ/Đen, trả về ``MatchSummary``.

LƯU Ý: Engine UCI/UCCI (Pikafish/Eleeye) cần file nhị phân ngoài. Nếu config
không cấu hình ``engine_path`` hoặc file không tồn tại, bài test engine sẽ bị
**bỏ qua an toàn** (trả ``None``) thay vì crash.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import torch

from src.env.graph_builder import RED
from src.env.xiangqi_env import XiangqiEnv
from src.evaluation.elo import MatchSummary, compute_elo, result_to_score
from src.training.reward import RESULT_DRAW, RESULT_LOSS, RESULT_WIN
from src.training.self_play import reason_to_result

logger = logging.getLogger(__name__)


class Agent(Protocol):
    """Giao diện chung cho mọi tác nhân (agent) cờ tướng."""

    name: str

    def select_action(self, state: dict) -> int:
        """Chọn hành động từ state (``{"graph", "legal_mask", "player", ...}``)."""
        ...

    def reset(self) -> None:
        """Reset trạng thái nội bộ trước mỗi ván (nếu cần)."""
        ...


@dataclass
class RandomAgent:
    """Tác nhân đi ngẫu nhiên trong tập nước hợp lệ (baseline yếu nhất)."""

    seed: int = 0
    name: str = "Random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """Không cần trạng thái nội bộ."""
        return None

    def select_action(self, state: dict) -> int:
        """Chọn ngẫu nhiên một hành động hợp lệ."""
        actions = state["legal_mask"].nonzero(as_tuple=False).flatten().tolist()
        if not actions:
            raise RuntimeError("RandomAgent không có nước hợp lệ để đi.")
        return int(self._rng.choice(actions))


class ModelAgent:
    """Tác nhân bọc mô hình ``XiangqiPolicyNet`` đã huấn luyện.

    Args:
        model: Mô hình đã nạp trọng số.
        device: Thiết bị chạy.
        deterministic: Dùng argmax thay vì lấy mẫu (mặc định True khi đánh giá).
        name: Tên hiển thị.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        name: str = "Model",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.name = name

    def reset(self) -> None:
        """Không cần trạng thái nội bộ."""
        return None

    @torch.no_grad()
    def select_action(self, state: dict) -> int:
        """Chọn hành động bằng mô hình (argmax hoặc lấy mẫu)."""
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([state["graph"]]).to(self.device)
        mask = state["legal_mask"].unsqueeze(0).to(self.device)
        actions, _, _, _ = self.model.act(
            batch, mask, deterministic=self.deterministic
        )
        return int(actions.item())


class EngineAgent:
    """Tác nhân bọc engine UCI/UCCI (Pikafish/Eleeye) qua ``cchess.engine``.

    Args:
        engine_path: Đường dẫn file nhị phân engine.
        protocol: ``"uci"`` (Pikafish) hoặc ``"ucci"`` (Eleeye).
        depth: Độ sâu tìm kiếm.
        threads: Số luồng engine.
        name: Tên hiển thị.

    Raises:
        FileNotFoundError: Nếu file engine không tồn tại.
    """

    def __init__(
        self,
        engine_path: str,
        protocol: str = "uci",
        depth: int = 2,
        threads: int = 1,
        name: str = "Engine",
    ) -> None:
        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file engine: {engine_path}")

        from cchess.engine import EngineManager

        self.name = name
        self.depth = depth
        self.manager = EngineManager()
        options = {"Threads": threads, "threads": threads}
        go_params = {"depth": depth}
        if protocol.lower() == "ucci":
            ok = self.manager.load_ucci(str(path), options, go_params)
        else:
            ok = self.manager.load_uci(str(path), options, go_params)
        if not ok:
            raise RuntimeError(f"Không nạp được engine: {engine_path}")
        logger.info("Đã nạp engine %s (protocol=%s, depth=%d)", path.name, protocol, depth)

    def reset(self) -> None:
        """Không cần trạng thái nội bộ."""
        return None

    def select_action(self, state: dict) -> int:
        """Hỏi engine nước tốt nhất, chuyển ICCS -> action index."""
        from src.env.constants import action_to_positions, positions_to_action
        from cchess.common import iccs2pos

        fen = state["fen"]
        action = self.manager.get_fen_score(fen)
        if not action or "move" not in action:
            # Engine không trả về nước đi -> chọn nước hợp lệ đầu tiên.
            legal = state["legal_mask"].nonzero(as_tuple=False).flatten().tolist()
            return int(legal[0]) if legal else 0

        iccs = action["move"]
        try:
            from_pos, to_pos = iccs2pos(iccs)
            return int(positions_to_action(from_pos, to_pos))
        except Exception:  # pragma: no cover - engine trả format lạ
            legal = state["legal_mask"].nonzero(as_tuple=False).flatten().tolist()
            return int(legal[0]) if legal else 0

    def close(self) -> None:
        """Đóng engine."""
        try:
            self.manager.quit()
        except Exception:  # pragma: no cover
            pass


def play_one_game(
    agent_red: Agent,
    agent_black: Agent,
    max_moves: int = 200,
    perpetual_check_limit: int = 6,
    legal_move_filter: bool = True,
    seed: int = 0,
) -> str:
    """Chơi một ván giữa hai agent, trả về kết quả theo góc nhìn **agent_red**.

    Args:
        agent_red: Agent cầm quân Đỏ.
        agent_black: Agent cầm quân Đen.
        max_moves: Số nước tối đa.
        perpetual_check_limit: Ngưỡng chiếu lặp.
        legal_move_filter: Lọc nước tự-chiếu-tướng.
        seed: Seed cho môi trường.

    Returns:
        ``"win"`` / ``"loss"`` / ``"draw"`` theo góc nhìn agent Đỏ.
    """
    agent_red.reset()
    agent_black.reset()

    env = XiangqiEnv(
        max_moves=max_moves,
        perpetual_check_limit=perpetual_check_limit,
        legal_move_filter=legal_move_filter,
        seed=seed,
    )
    state = env.reset()

    while True:
        player = state["player"]
        agent = agent_red if player == RED else agent_black
        action = agent.select_action(state)

        state, _reward, done, info = env.step(action)
        if done:
            # Bên vừa đi là bên vừa thắng nếu checkmate; suy ra kết quả theo Đỏ.
            red_just_moved = player == RED
            result_red = reason_to_result(info["reason"], red_just_moved)
            return result_red or RESULT_DRAW


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int = 100,
    max_moves: int = 200,
    perpetual_check_limit: int = 6,
    legal_move_filter: bool = True,
    opponent_elo: float = 0.0,
    elo_k: float = 32.0,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    label: str = "",
) -> MatchSummary:
    """Chạy ``num_games`` ván giữa A và B, xen kẽ màu, tổng hợp kết quả.

    Args:
        agent_a: Agent được đánh giá (kết quả tính theo góc nhìn A).
        agent_b: Đối thủ.
        num_games: Số ván.
        max_moves: Số nước tối đa mỗi ván.
        perpetual_check_limit: Ngưỡng chiếu lặp.
        legal_move_filter: Lọc nước tự-chiếu-tướng.
        opponent_elo: Elo giả định của đối thủ (dùng để tính Elo của A).
        elo_k: Hệ số K cho Elo.
        bootstrap_samples: Số mẫu bootstrap cho CI.
        seed: Seed chung.
        label: Nhãn hiển thị.

    Returns:
        ``MatchSummary`` với số thắng/hoà/thua và Elo + CI.
    """
    rng = random.Random(seed)
    wins = draws = losses = 0
    scores: List[float] = []
    opponents: List[float] = []

    for game_idx in range(num_games):
        # Xen kẽ: ván chẵn A cầm Đỏ, ván lẻ A cầm Đen.
        a_plays_red = game_idx % 2 == 0
        game_seed = rng.randint(0, 2**31 - 1)
        if a_plays_red:
            result_a = play_one_game(
                agent_a, agent_b, max_moves, perpetual_check_limit,
                legal_move_filter, game_seed,
            )
        else:
            result_b = play_one_game(
                agent_b, agent_a, max_moves, perpetual_check_limit,
                legal_move_filter, game_seed,
            )
            # Đảo ngược kết quả vì kết quả tính theo bên cầm Đỏ (= agent_b).
            result_a = {
                RESULT_WIN: RESULT_LOSS,
                RESULT_LOSS: RESULT_WIN,
                RESULT_DRAW: RESULT_DRAW,
            }[result_b]

        if result_a == RESULT_WIN:
            wins += 1
        elif result_a == RESULT_LOSS:
            losses += 1
        else:
            draws += 1

        scores.append(result_to_score(result_a))
        opponents.append(opponent_elo)

    elo, ci_low, ci_high = compute_elo(
        scores, opponents, k=elo_k,
        bootstrap_samples=bootstrap_samples, seed=seed,
    )
    summary = MatchSummary(
        games=num_games, wins=wins, draws=draws, losses=losses,
        elo=elo, ci_low=ci_low, ci_high=ci_high,
    )
    logger.info(
        "%s: %d ván | T/B/H %d/%d/%d | winrate %.3f | Elo %.1f [%.1f, %.1f]",
        label or agent_a.name, num_games, wins, draws, losses,
        summary.win_rate, elo, ci_low, ci_high,
    )
    return summary


