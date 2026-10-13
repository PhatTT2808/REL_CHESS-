"""Kiểm thử tầng môi trường: reset / step / legal mask / trạng thái kết thúc."""

from __future__ import annotations

import random

import pytest
import torch

from src.env import XiangqiEnv
from src.env.constants import ACTION_SPACE, action_to_positions, positions_to_action
from src.env.xiangqi_env import (
    REASON_CHECKMATE,
    REASON_KING_CAPTURED,
    REASON_MAX_MOVES,
)


def test_action_encoding_roundtrip() -> None:
    """Mã hóa/giải mã action phải đối xứng (bijection)."""
    from_pos, to_pos = (1, 9), (2, 7)
    action = positions_to_action(from_pos, to_pos)
    assert 0 <= action < ACTION_SPACE
    # idx = y*9 + x: (1,9) -> 9*9+1 = 82 ; (2,7) -> 7*9+2 = 65
    assert action == 82 * 90 + 65
    back_from, back_to = action_to_positions(action)
    assert back_from == from_pos
    assert back_to == to_pos


def test_reset_returns_initial_state(env: XiangqiEnv) -> None:
    """``reset`` trả về đúng cấu trúc state và thế cờ khai cuộc."""
    state = env.reset()
    assert set(["fen", "graph", "legal_mask", "player"]).issubset(state.keys())
    assert state["fen"].startswith("rnbakabnr")
    assert state["graph"].x.shape[0] == 32  # 32 quân ban đầu
    assert state["graph"].x.shape[1] == 10
    assert state["player"] == 1  # Đỏ đi trước


def test_legal_mask_shape_and_content(env: XiangqiEnv) -> None:
    """Legal mask phải là BoolTensor(8100) và có đúng số nước khai cuộc."""
    state = env.reset()
    mask = state["legal_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (ACTION_SPACE,)
    assert mask.dtype == torch.bool
    n_legal = int(mask.sum())
    assert 0 < n_legal < ACTION_SPACE
    assert n_legal == 44  # thế khai cuộc cờ tướng có 44 nước hợp lệ


def test_legal_mask_excludes_self_check(env: XiangqiEnv) -> None:
    """Mọi nước trong mask phải KHÔNG để lộ tướng mình (đúng luật)."""
    env.reset()
    for action in env.legal_actions()[:20]:
        from_pos, to_pos = action_to_positions(action)
        assert env.board.get_fench(from_pos) is not None
        assert env.board.is_valid_move(from_pos, to_pos)
        assert not env.board.is_checked_move(from_pos, to_pos)


def test_step_advances_turn_and_updates_state(env: XiangqiEnv) -> None:
    """Sau ``step``, lượt đổi và state được cập nhật."""
    state = env.reset()
    player_before = state["player"]
    action = int(state["legal_mask"].nonzero()[0])

    new_state, reward, done, info = env.step(action)
    assert new_state["player"] != player_before
    assert env.move_count == 1
    assert isinstance(reward, float)
    assert "reason" in info
    if not done:
        assert reward == pytest.approx(0.0)


def test_illegal_action_raises(env: XiangqiEnv) -> None:
    """Hành động nằm ngoài legal mask phải raise ``ValueError``."""
    state = env.reset()
    mask = state["legal_mask"]
    illegal = int((~mask).nonzero()[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_checkmate_terminates_game() -> None:
    """Thế chiếu bí phải kết thúc ván với reward +1 cho bên vừa đi."""
    env = XiangqiEnv(max_moves=200, legal_move_filter=False, seed=0)
    env.reset(fen="3k5/9/9/9/9/3R5/9/9/4R4/4K4 w")

    # Tìm nước của Đỏ dẫn tới chiếu bí Đen.
    mate_action = None
    for action in env.legal_actions():
        (x1, y1), (x2, y2) = action_to_positions(action)
        trial = env.board.copy()
        trial.move((x1, y1), (x2, y2))
        if trial.is_checkmate() or trial.no_moves():
            mate_action = action
            break

    assert mate_action is not None, "Không tìm được nước chiếu bí trong thế cờ test."
    _state, reward, done, info = env.step(mate_action)
    assert done is True
    assert reward == pytest.approx(1.0)
    assert info["reason"] == REASON_CHECKMATE


def test_max_moves_ends_as_draw() -> None:
    """Chạm trần số nước phải kết thúc ván với reward 0 (hoà)."""
    env = XiangqiEnv(max_moves=4, legal_move_filter=False, seed=3)
    state = env.reset()
    rng = random.Random(3)
    done = False
    reward = None
    info = {}
    for _ in range(10):
        actions = state["legal_mask"].nonzero(as_tuple=False).flatten().tolist()
        action = rng.choice(actions)
        state, reward, done, info = env.step(action)
        if done:
            break
    assert done is True
    assert info["reason"] == REASON_MAX_MOVES
    assert reward == pytest.approx(0.0)


def test_king_captured_terminates_cleanly() -> None:
    """Khi tướng bị ăn, env không crash và báo kết thúc hợp lệ."""
    env = XiangqiEnv(max_moves=200, legal_move_filter=False, seed=0)
    env.reset(fen="3k5/9/9/9/9/9/9/4r4/9/4K4 w")

    # Bên Đen (xe ở e2) có thể ăn tướng Đỏ ở e0.
    action = None
    for a in env.legal_actions():
        (x1, y1), (x2, y2) = action_to_positions(a)
        target = env.board.get_fench((x2, y2))
        if target is not None and target.lower() == "k":
            action = a
            break
    # Nếu nước ăn tướng thuộc bên Đen (đi sau), ta cho Đỏ đi nước đầu rồi kiểm tra.
    if action is None:
        env.step(env.legal_actions()[0])
        for a in env.legal_actions():
            (x1, y1), (x2, y2) = action_to_positions(a)
            target = env.board.get_fench((x2, y2))
            if target is not None and target.lower() == "k":
                action = a
                break
    assert action is not None, "Không tìm được nước ăn tướng trong thế cờ test."
    _s, reward, done, info = env.step(action)
    assert done is True
    assert info["reason"] == REASON_KING_CAPTURED
    assert reward == pytest.approx(1.0)
    # Sau khi mất tướng, mask phải rỗng và không crash.
    assert int(env.get_legal_mask().sum()) == 0


def test_graph_builder_shapes(env: XiangqiEnv) -> None:
    """Đồ thị phải có shape đúng và chứa đủ 3 loại cạnh."""
    state = env.reset()
    data = state["graph"]
    n = data.x.shape[0]
    assert n == 32
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 3
    assert data.edge_index.shape[1] == data.edge_attr.shape[0]
    assert data.node_pos.shape == (n, 2)
    edge_types = set(data.edge_attr.argmax(dim=1).tolist())
    assert {0, 1, 2}.issubset(edge_types)


def test_render_returns_text(env: XiangqiEnv) -> None:
    """``render`` trả về chuỗi chứa FEN, không in ra stdout."""
    env.reset()
    text = env.render()
    assert isinstance(text, str)
    assert "FEN:" in text
    assert "rnbakabnr" in text


def test_env_determinism_with_same_seed() -> None:
    """Hai env cùng seed phải cho cùng chuỗi hành động."""

    def rollout() -> list:
        e = XiangqiEnv(max_moves=30, seed=99)
        e.reset()
        rng = random.Random(99)
        seq = []
        for _ in range(10):
            acts = e.legal_actions()
            a = rng.choice(acts)
            seq.append(a)
            _s, _r, done, _i = e.step(a)
            if done:
                break
        return seq

    assert rollout() == rollout()

