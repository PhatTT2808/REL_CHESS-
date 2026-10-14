"""Sinh dữ liệu self-play với đối thủ từ Opponent Pool (PFSP).

Ý tưởng:
    - Agent hiện tại đấu với **các checkpoint cũ** trong pool, thay vì chỉ tự
      đấu với chính mình (tránh "quên" chiến thuật cũ và tăng tính đa dạng).
    - **PFSP** (Prioritized Fictitious Self-Play): ưu tiên đối thủ mà agent
      thắng khoảng 50% (vừa sức), theo công thức
      ``p ∝ (1 - |2 * win_rate - 1|) ** k``.

Mỗi ván sinh ra một ``Trajectory`` chứa đúng dữ liệu PPO cần.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.env.graph_builder import RED
from src.env.xiangqi_env import (
    REASON_CHECKMATE,
    REASON_KING_CAPTURED,
    REASON_MAX_MOVES,
    REASON_ONGOING,
    REASON_PERPETUAL,
    REASON_REPETITION,
    XiangqiEnv,
)
from src.training.ppo import PPOAgent, Trajectory
from src.training.reward import (
    RESULT_DRAW,
    RESULT_LOSS,
    RESULT_WIN,
    RewardConfig,
    compute_reward,
    material_potential,  # giữ lại cho _terminal_value nếu cần
)
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ensure_dir
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def reason_to_result(reason: str, mover_is_winner_side: bool) -> Optional[str]:
    """Chuyển lý do kết thúc ván thành kết cục theo góc nhìn một bên.

    Args:
        reason: Lý do từ ``XiangqiEnv`` (vd ``"checkmate"``).
        mover_is_winner_side: ``True`` nếu bên đang xét là **bên vừa đi**
            tại thời điểm ván kết thúc.

    Returns:
        ``"win"`` / ``"loss"`` / ``"draw"`` / ``None`` (chưa kết thúc).
    """
    if reason == REASON_ONGOING:
        return None
    if reason in (REASON_CHECKMATE, REASON_KING_CAPTURED):
        # Bên vừa đi là bên thắng.
        return RESULT_WIN if mover_is_winner_side else RESULT_LOSS
    if reason in (REASON_REPETITION, REASON_PERPETUAL, REASON_MAX_MOVES):
        return RESULT_DRAW
    return RESULT_DRAW


@dataclass
class OpponentStat:
    """Thống kê đối thủ trong pool (phục vụ PFSP)."""

    path: Path
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0

    @property
    def win_rate(self) -> float:
        """Tỷ lệ thắng (bỏ qua hoà) của agent chính trước đối thủ này."""
        decided = self.wins + self.losses
        if decided == 0:
            return 0.5
        return self.wins / decided

    def update(self, result: str) -> None:
        """Cập nhật thống kê sau một ván."""
        self.games += 1
        if result == RESULT_WIN:
            self.wins += 1
        elif result == RESULT_LOSS:
            self.losses += 1
        else:
            self.draws += 1


class OpponentPool:
    """Bể đối thủ gồm các checkpoint cũ + model ngẫu nhiên ban đầu.

    Args:
        pool_dir: Thư mục lưu checkpoint đối thủ (từ config ``paths.pool``).
        model_kwargs: Tham số khởi tạo ``XiangqiPolicyNet`` để dựng đối thủ.
        device: Thiết bị chạy đối thủ.
        pfsp_power: Số mũ ``k`` trong công thức PFSP (càng lớn càng ưu tiên
            đối thủ vừa sức).
    """

    def __init__(
        self,
        pool_dir: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: torch.device | str = "cpu",
        pfsp_power: float = 2.0,
    ) -> None:
        self.pool_dir = ensure_dir(pool_dir)
        self.model_kwargs = model_kwargs or {}
        self.device = torch.device(device)
        self.pfsp_power = pfsp_power
        self.stats: List[OpponentStat] = []
        self._refresh()

    def _refresh(self) -> None:
        """Quét thư mục pool để cập nhật danh sách checkpoint."""
        existing = {s.path for s in self.stats}
        for path in sorted(self.pool_dir.glob("step_*.pt")):
            if path not in existing:
                self.stats.append(OpponentStat(path=path))

    def __len__(self) -> int:
        return len(self.stats)

    def add_snapshot(self, model: torch.nn.Module, step: int) -> Path:
        """Lưu bản snapshot của model hiện tại vào pool.

        Args:
            model: Model hiện tại (agent chính).
            step: Bước huấn luyện.

        Returns:
            Đường dẫn checkpoint đã lưu.
        """
        path = self.pool_dir / f"step_{step}.pt"
        torch.save({"step": step, "model_state": model.state_dict()}, path)
        self.stats.append(OpponentStat(path=path))
        logger.info("Đã thêm đối thủ vào pool: %s", path)
        return path

    def sample(self, rng: random.Random) -> OpponentStat:
        """Chọn một đối thủ theo PFSP (ưu tiên tỷ lệ thắng ~50%).

        Nếu pool rỗng sau khi lọc, tự tạo một đối thủ "khởi tạo ngẫu nhiên"
        bằng cách thêm snapshot rỗng — bảo đảm self-play luôn chạy được.
        """
        self._refresh()
        if not self.stats:
            logger.warning(
                "Pool rỗng (có thể do checkpoint không khớp kiến trúc). "
                "Tạo đối thủ mới từ trọng số ngẫu nhiên."
            )
            from src.models.policy_net import XiangqiPolicyNet

            fresh = XiangqiPolicyNet(**self.model_kwargs)
            self.add_snapshot(fresh, step=0)
            return self.stats[-1]
        if len(self.stats) == 1:
            return self.stats[0]

        weights = [
            max(1e-3, (1.0 - abs(2.0 * s.win_rate - 1.0)) ** self.pfsp_power)
            for s in self.stats
        ]
        total = sum(weights)
        probs = [w / total for w in weights]
        return rng.choices(self.stats, weights=probs, k=1)[0]

    def load_opponent(self, stat: OpponentStat) -> torch.nn.Module:
        """Nạp model đối thủ từ checkpoint (không cần optimizer).

        Nếu checkpoint có kiến trúc khác với config hiện tại (ví dụ pool chứa
        snapshot từ lần chạy với hyperparameter khác), hàm sẽ **bỏ qua an toàn**
        và trả về một model khởi tạo ngẫu nhiên thay vì crash.
        """
        from src.models.policy_net import XiangqiPolicyNet

        model = XiangqiPolicyNet(**self.model_kwargs)
        try:
            load_checkpoint(stat.path, model, map_location=str(self.device))
        except RuntimeError as exc:
            logger.warning(
                "Bỏ qua checkpoint đối thủ %s (kiến trúc không khớp): %s",
                stat.path.name, str(exc).splitlines()[0],
            )
            # Loại khỏi pool để không thử lại nhiều lần.
            self.stats = [s for s in self.stats if s.path != stat.path]
        model.to(self.device).eval()
        return model

    def record(self, stat: OpponentStat, result: str) -> None:
        """Ghi nhận kết quả một ván vào thống kê PFSP."""
        stat.update(result)


@dataclass
class SelfPlayStats:
    """Thống kê một batch self-play."""

    games: int = 0
    steps: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_moves: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_moves(self) -> float:
        """Số nước trung bình mỗi ván."""
        return self.total_moves / self.games if self.games else 0.0


class SelfPlayRunner:
    """Sinh batch trajectory bằng cách cho agent đấu với Opponent Pool.

    Args:
        agent: ``PPOAgent`` của agent chính (đang được huấn luyện).
        pool: ``OpponentPool``.
        reward_config: Hệ số phần thưởng.
        max_moves: Số nước tối đa mỗi ván.
        perpetual_check_limit: Ngưỡng chiếu lặp.
        legal_move_filter: Lọc nước tự-chiếu-tướng.
        model_kwargs: Tham số dựng model đối thủ.
        device: Thiết bị.
        seed: Seed cho RNG nội bộ.
    """

    def __init__(
        self,
        agent: PPOAgent,
        pool: OpponentPool,
        reward_config: Optional[RewardConfig] = None,
        max_moves: int = 200,
        perpetual_check_limit: int = 6,
        legal_move_filter: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.agent = agent
        self.pool = pool
        self.reward_config = reward_config or RewardConfig()
        self.max_moves = max_moves
        self.perpetual_check_limit = perpetual_check_limit
        self.legal_move_filter = legal_move_filter
        self.model_kwargs = model_kwargs or {}
        self.device = torch.device(device)
        self.rng = random.Random(seed)

    def generate_batch(
        self, num_games: int, record_stats: bool = True
    ) -> Tuple[List[Trajectory], SelfPlayStats]:
        """Sinh ``num_games`` ván, trả về trajectory và thống kê.

        Agent chính đi xen kẽ Đỏ/Đen để không thiên vị một bên.

        Args:
            num_games: Số ván cần sinh.
            record_stats: Cập nhật thống kê PFSP cho pool.

        Returns:
            ``(trajectories, stats)``.
        """
        stats = SelfPlayStats()
        trajectories: List[Trajectory] = []

        for game_idx in range(num_games):
            agent_plays_red = (game_idx % 2 == 0)
            traj = self._play_one_game(agent_plays_red, stats, record_stats)
            if traj is not None and len(traj) > 0:
                trajectories.append(traj)

        logger.debug(
            "Self-play batch: %d ván, %d bước, T/B/H = %d/%d/%d",
            stats.games, stats.steps, stats.wins, stats.losses, stats.draws,
        )
        return trajectories, stats

    def _play_one_game(
        self, agent_plays_red: bool, stats: SelfPlayStats, record_stats: bool
    ) -> Optional[Trajectory]:
        """Chơi một ván giữa agent chính (đang học) và một đối thủ từ pool.

        Chỉ ghi các bước của **agent chính** vào trajectory (đối thủ không học).
        Reward gán theo góc nhìn bên vừa đi = agent chính.
        """
        opponent_stat = self.pool.sample(self.rng)
        opponent = self.pool.load_opponent(opponent_stat)

        agent_color = RED if agent_plays_red else (3 - RED)  # RED=1, BLACK=2
        env = XiangqiEnv(
            max_moves=self.max_moves,
            perpetual_check_limit=self.perpetual_check_limit,
            legal_move_filter=self.legal_move_filter,
            seed=self.rng.randint(0, 2**31 - 1),
        )
        state = env.reset()
        traj = Trajectory()
        result_for_agent: Optional[str] = RESULT_DRAW
        reason = REASON_ONGOING

        while True:
            player = state["player"]
            graph = state["graph"]
            mask = state["legal_mask"]
            agent_just_moved = player == agent_color

            if agent_just_moved:
                action, log_prob, value, attn = self.agent.collect_step(graph, mask)
            else:
                action = self._opponent_action(opponent, graph, mask)
                log_prob, value, attn = 0.0, 0.0, None

            # Phi(s) TRƯỚC nước đi. Phải tính trước vì ``board.move()`` của
            # cchess sửa bàn cờ TẠI CHỖ (không trả về bản sao), nên giữ tham chiếu
            # ``env.board`` rồi tính sau sẽ cho cùng giá trị với s' => shaping = 0.
            phi_prev = material_potential(
                env.board, agent_color, self.reward_config.material_coef
            )
            next_state, _env_reward, done, info = env.step(action)
            reason = info["reason"]

            if agent_just_moved:
                if done:
                    r = self._terminal_value(reason, mover_is_winner_side=True)
                else:
                    r = compute_reward(
                        env.board, agent_color, config=self.reward_config,
                        terminal=False, phi_prev=phi_prev,
                    )
                traj.append(
                    graph=graph,
                    legal_mask=mask,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=r,
                    done=done,
                    player=player,
                    attn_weights=attn,
                )

            if done:
                result_for_agent = reason_to_result(reason, agent_just_moved)
                traj.result = result_for_agent
                traj.reason = reason
                break

            state = next_state

        if record_stats and result_for_agent is not None:
            self.pool.record(opponent_stat, result_for_agent)

        stats.games += 1
        stats.steps += len(traj)
        stats.total_moves += env.move_count
        stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
        if result_for_agent == RESULT_WIN:
            stats.wins += 1
        elif result_for_agent == RESULT_LOSS:
            stats.losses += 1
        else:
            stats.draws += 1

        return traj

    def _opponent_action(self, opponent: torch.nn.Module, graph, mask) -> int:
        """Đối thủ chọn hành động (lấy mẫu để tạo đa dạng chiến thuật)."""
        from torch_geometric.data import Batch

        with torch.no_grad():
            batch = Batch.from_data_list([graph]).to(self.device)
            m = mask.unsqueeze(0).to(self.device)
            actions, _, _, _ = opponent.act(batch, m, deterministic=False)
            return int(actions.item())

    def _terminal_value(self, reason: str, mover_is_winner_side: bool) -> float:
        """Phần thưởng kết cục theo góc nhìn bên vừa đi."""
        result = reason_to_result(reason, mover_is_winner_side)
        if result == RESULT_WIN:
            return self.reward_config.win
        if result == RESULT_LOSS:
            return self.reward_config.loss
        return self.reward_config.draw



