"""Kiểm thử phần thưởng — đặc biệt là potential-based shaping.

Bối cảnh (sau vòng train 1e6 bước đầu tiên):
    - Shaping kiểu cũ (``coef * Phi(s)`` mỗi bước) cộng dồn tới ~+12.5 điểm/ván,
      gấp 12 lần terminal reward -> agent học "ăn quân cho nhiều rồi giằng co"
      thay vì chiếu hết (91/100 ván hoà).
    - Value head ``tanh`` bị kẹp trong [-1,1] nên không biểu diễn được return.

Các test dưới đây chốt lại hành vi ĐÚNG sau khi sửa.
"""

from __future__ import annotations

import pytest

from src.env import XiangqiEnv
from src.env.constants import action_to_positions
from src.env.graph_builder import RED
from src.training.reward import (
    RESULT_DRAW,
    RESULT_LOSS,
    RESULT_WIN,
    RewardConfig,
    compute_reward,
    material_potential,
    potential_shaping,
    terminal_reward,
)


# ----------------------------------------------------------------------
# Potential function & potential-based shaping
# ----------------------------------------------------------------------
def test_material_potential_zero_on_starting_position() -> None:
    """Thế cờ khai cuộc cân bằng quân -> Phi(s) = 0 cho cả hai bên."""
    env = XiangqiEnv(max_moves=200, seed=1)
    env.reset()
    assert material_potential(env.board, RED, 0.01) == pytest.approx(0.0, abs=1e-9)
    assert material_potential(env.board, 2, 0.01) == pytest.approx(0.0, abs=1e-9)


def test_material_potential_zero_when_coef_zero() -> None:
    """Tắt shaping bằng cách đặt ``material_coef = 0``."""
    env = XiangqiEnv(max_moves=200, seed=1)
    env.reset()
    assert material_potential(env.board, RED, 0.0) == 0.0


def test_potential_shaping_telescopes_over_game() -> None:
    """Tổng potential-based shaping chỉ phụ thuộc trạng thái ĐẦU và CUỐI.

    Đây là tính chất bảo toàn chính sách tối ưu của Ng et al. (1999): tổng
    ``gamma*Phi(s') - Phi(s)`` dọc ván = ``gamma^T*Phi(s_T) - Phi(s_0)``, KHÔNG
    cộng dồn theo độ dài ván (khác với cách cũ cộng mỗi bước một lần).
    """
    gamma = 0.99
    env = XiangqiEnv(max_moves=200, seed=3)
    state = env.reset()
    phi0 = material_potential(env.board, RED, 0.01)
    total = 0.0
    steps = 0
    while True:
        is_model = state["player"] == RED
        phi_prev = material_potential(env.board, RED, 0.01)
        action = int(state["legal_mask"].nonzero()[0])
        state, _r, done, _info = env.step(action)
        if is_model and not done:
            phi_next = material_potential(env.board, RED, 0.01)
            total += potential_shaping(phi_prev, phi_next, gamma=gamma)
            steps += 1
        if done:
            break
    # Tổng shaping phải nhỏ (bounded), không cỡ hàng chục điểm.
    assert steps > 0
    assert abs(total) < 1.0, f"Tổng shaping bị tích luỹ: {total}"
    assert phi0 == pytest.approx(0.0, abs=1e-9)


def test_potential_shaping_is_zero_when_no_material_change() -> None:
    """Nếu trạng thái s' không đổi chênh lệch quân thì shaping = 0."""
    env = XiangqiEnv(max_moves=200, seed=5)
    env.reset()
    phi = material_potential(env.board, RED, 0.01)
    assert potential_shaping(phi, phi, gamma=0.99) == pytest.approx(0.0, abs=1e-9)


def test_compute_reward_terminal_uses_only_outcome() -> None:
    """Ván kết thúc: chỉ trả terminal reward, không cộng shaping/penalty."""
    env = XiangqiEnv(max_moves=200, seed=7)
    env.reset()
    cfg = RewardConfig()
    assert compute_reward(env.board, RED, RESULT_WIN, cfg, terminal=True) == 1.0
    assert compute_reward(env.board, RED, RESULT_LOSS, cfg, terminal=True) == -1.0
    assert compute_reward(env.board, RED, RESULT_DRAW, cfg, terminal=True) == pytest.approx(-0.1)


def test_compute_reward_includes_step_penalty() -> None:
    """Nước đi giữa ván luôn bị trừ ``step_penalty`` để chống kéo giờ."""
    env = XiangqiEnv(max_moves=200, seed=9)
    env.reset()
    cfg = RewardConfig(step_penalty=0.01, material_coef=0.0)
    phi_prev = material_potential(env.board, RED, 0.0)  # = 0.0
    r = compute_reward(env.board, RED, config=cfg, terminal=False, phi_prev=phi_prev)
    # Không đổi quân + phạt bước -> đúng bằng -step_penalty.
    assert r == pytest.approx(-0.01)


def test_total_reward_terminal_dominates_shaping() -> None:
    """Tổng shaping + terminal phải còn cùng thang với terminal (±1).

    Đây là hồi quy cho lỗi cũ (shaping ~12 điểm >> terminal 1 điểm): với
    potential-based shaping + step_penalty, tổng reward mỗi bước phải nhỏ.
    """
    cfg = RewardConfig()
    env = XiangqiEnv(max_moves=200, seed=11)
    state = env.reset()
    rewards = []
    while True:
        is_model = state["player"] == RED
        phi_prev = material_potential(env.board, RED, cfg.material_coef)
        action = int(state["legal_mask"].nonzero()[0])
        state, _r, done, _info = env.step(action)
        if is_model:
            if done:
                rewards.append(terminal_reward(RESULT_DRAW, cfg))
            else:
                rewards.append(
                    compute_reward(
                        env.board, RED, config=cfg,
                        terminal=False, phi_prev=phi_prev,
                    )
                )
        if done:
            break
    max_abs = max(abs(r) for r in rewards)
    # Mỗi bước reward không vượt quá ~step_penalty + biến động material nhỏ.
    assert max_abs < 0.5, f"Reward từng bước quá lớn: {max_abs}"
    assert sum(rewards) > -10.0


def test_action_to_positions_roundtrip_in_reward_test() -> None:
    """Bảo đảm helper mã hoá hành động dùng được trong test (idx = y*9 + x)."""
    a = 82 * 90 + 65
    # 82 -> y=9, x=1 ;  65 -> y=7, x=2
    assert action_to_positions(a) == ((1, 9), (2, 7))


def test_reward_config_from_dict_defaults() -> None:
    """Đọc reward từ YAML (hoặc dict) phải dùng đúng giá trị mặc định mới."""
    cfg = RewardConfig.from_dict({})
    assert cfg.draw == pytest.approx(-0.1)
    assert cfg.step_penalty == pytest.approx(0.005)
    assert cfg.material_coef == pytest.approx(0.01)
