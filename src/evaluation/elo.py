"""Tính Elo cho agent cờ tướng, kèm khoảng tin cậy (bootstrap CI).

Công thức Elo chuẩn:
    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))       # điểm kỳ vọng của A
    R_A' = R_A + K * (S_A - E_A)                    # cập nhật sau ván
với ``S_A`` là điểm thực tế (thắng=1, hoà=0.5, thua=0).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

WIN: float = 1.0
DRAW: float = 0.5
LOSS: float = 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Điểm kỳ vọng của A khi đấu với B (thang Elo 400)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = 32.0
) -> Tuple[float, float]:
    """Cập nhật Elo của cả hai bên sau một ván.

    Args:
        rating_a: Elo hiện tại của A.
        rating_b: Elo hiện tại của B.
        score_a: Điểm thực tế của A (1 thắng, 0.5 hoà, 0 thua).
        k: Hệ số K.

    Returns:
        ``(new_rating_a, new_rating_b)``.
    """
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - exp_a))
    return new_a, new_b


def result_to_score(result: str) -> float:
    """Chuyển kết quả ván (``"win"``/``"draw"``/``"loss"``) thành điểm."""
    return {"win": WIN, "draw": DRAW, "loss": LOSS}.get(result, DRAW)


def compute_elo(
    scores: Sequence[float],
    opponent_ratings: Sequence[float],
    k: float = 32.0,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Ước lượng Elo của agent từ một loạt ván đấu với các đối thủ đã biết Elo.

    Cách làm: xuất phát từ Elo 0, cập nhật tuần tự theo từng ván (kiểu "online"),
    sau đó dùng **bootstrap** (lấy mẫu lại có hoàn lại) để suy ra khoảng tin cậy.

    Args:
        scores: Điểm từng ván theo góc nhìn agent (1/0.5/0).
        opponent_ratings: Elo của đối thủ trong từng ván (cùng độ dài).
        k: Hệ số K.
        bootstrap_samples: Số lần lấy mẫu bootstrap.
        seed: Seed cho bootstrap (tái lập được).
        confidence: Mức tin cậy (mặc định 0.95 -> khoảng 2.5%..97.5%).

    Returns:
        ``(elo, ci_low, ci_high)``.

    Raises:
        ValueError: Nếu số ván bằng 0 hoặc độ dài hai danh sách khác nhau.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cần ít nhất 1 ván để tính Elo.")
    if len(opponent_ratings) != n:
        raise ValueError("scores và opponent_ratings phải cùng độ dài.")

    def _estimate(pairs: Sequence[Tuple[float, float]]) -> float:
        rating = 0.0
        for score, opp in pairs:
            exp = expected_score(rating, opp)
            rating += k * (score - exp)
        return rating

    pairs = list(zip(scores, opponent_ratings))
    elo = _estimate(pairs)

    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(bootstrap_samples):
        resampled = [pairs[rng.randrange(n)] for _ in range(n)]
        samples.append(_estimate(resampled))
    samples.sort()

    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, int(alpha * bootstrap_samples) - 1)
    high_idx = min(bootstrap_samples - 1, int((1.0 - alpha) * bootstrap_samples))
    return elo, samples[low_idx], samples[high_idx]


@dataclass
class MatchSummary:
    """Tổng kết một loạt ván đấu."""

    games: int
    wins: int
    draws: int
    losses: int
    elo: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    @property
    def score_rate(self) -> float:
        """Tỷ lệ điểm tổng (win + 0.5*draw) / games."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def win_rate(self) -> float:
        """Tỷ lệ thắng."""
        return self.wins / self.games if self.games else 0.0

    def as_row(self, label: str = "") -> str:
        """Định dạng một dòng bảng kết quả (markdown)."""
        elo_str = (
            f"{self.elo:7.1f} [{self.ci_low:7.1f}, {self.ci_high:7.1f}]"
            if self.elo is not None
            else "     n/a"
        )
        return (
            f"| {label:<28} | {self.games:>5} | {self.wins:>5} | {self.draws:>5} | "
            f"{self.losses:>6} | {self.win_rate:>8.3f} | {self.score_rate:>9.3f} | "
            f"{elo_str} |"
        )

    @staticmethod
    def table_header() -> str:
        """Header của bảng kết quả (markdown)."""
        return (
            "| Đối thủ                      | Games |  Win  |  Draw |   Loss | "
            "WinRate  | ScoreRate | Elo [CI 95%]              |"
        )
