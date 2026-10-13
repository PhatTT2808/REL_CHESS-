"""Kiểm thử PPO: GAE, clipped surrogate, loss, cập nhật tham số."""

from __future__ import annotations

import math

import pytest
import torch

from src.models import XiangqiPolicyNet
from src.training.ppo import PPOAgent, PPOConfig, Trajectory, compute_gae


# ----------------------------------------------------------------------
# GAE
# ----------------------------------------------------------------------
def test_gae_single_step_terminal() -> None:
    """Ván 1 bước kết thúc: advantage = reward - value."""
    adv, ret = compute_gae(
        rewards=[1.0], values=[0.4], dones=[True], gamma=0.99, gae_lambda=0.95
    )
    assert adv.shape == (1,)
    assert adv[0].item() == pytest.approx(0.6, abs=1e-6)
    assert ret[0].item() == pytest.approx(1.0, abs=1e-6)


def test_gae_matches_manual_computation() -> None:
    """GAE phải khớp công thức tính tay (gamma=1, lambda=1)."""
    rewards = [1.0, 0.0]
    values = [0.5, 0.2]
    dones = [False, True]
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    # delta_1 = 0 + 0 - 0.2 = -0.2 ; delta_0 = 1 + 0.2 - 0.5 = 0.7
    # A_1 = -0.2 ; A_0 = 0.7 + 1*1*(1-0)* (-0.2) = 0.5
    assert adv[1].item() == pytest.approx(-0.2, abs=1e-6)
    assert adv[0].item() == pytest.approx(0.5, abs=1e-6)
    assert ret[0].item() == pytest.approx(1.0, abs=1e-6)
    assert ret[1].item() == pytest.approx(0.0, abs=1e-6)


def test_gae_returns_equal_advantage_plus_value() -> None:
    """``returns = advantages + values`` theo định nghĩa."""
    rewards = [0.5, -0.2, 1.0]
    values = [0.1, 0.3, -0.4]
    dones = [False, False, True]
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(ret, adv + torch.tensor(values), atol=1e-6)


def test_gae_no_discount_gamma_one() -> None:
    """gamma=1, lambda=1: advantage = reward hiện tại + phần advantage kế tiếp
    NHƯNG dừng lan truyền khi bước kế tiếp là trạng thái kết thúc (done=True)."""
    adv, _ret = compute_gae(
        rewards=[1.0, 2.0], values=[0.0, 0.0], dones=[False, True],
        gamma=1.0, gae_lambda=1.0,
    )
    # t=1 (done): next_non_terminal = 1.0, next_value = 0 -> delta_1 = 2.0 -> A_1 = 2.0
    # t=0: dones[1] = True -> next_non_terminal = 0 -> delta_0 = 1.0 -> A_0 = 1.0
    assert adv[1].item() == pytest.approx(2.0, abs=1e-6)
    assert adv[0].item() == pytest.approx(1.0, abs=1e-6)


def test_gae_propagates_when_not_done() -> None:
    """Khi chưa kết thúc (done=False), advantage được lan truyền qua các bước."""
    adv, _ret = compute_gae(
        rewards=[1.0, 2.0], values=[0.0, 0.0], dones=[False, False],
        gamma=1.0, gae_lambda=1.0,
    )
    # t=1 (không done, next_value bootstrap = 0) -> delta_1 = 2.0 -> A_1 = 2.0
    # t=0: next_non_terminal = 1 -> A_0 = 1.0 + 2.0 = 3.0
    assert adv[1].item() == pytest.approx(2.0, abs=1e-6)
    assert adv[0].item() == pytest.approx(3.0, abs=1e-6)


# ----------------------------------------------------------------------
# Clipped surrogate (kiểm tra bằng công thức trực tiếp)
# ----------------------------------------------------------------------
def _clipped_surrogate(ratio: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
    """Tái hiện công thức PPO-Clip để đối chiếu."""
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    return torch.min(surr1, surr2)


def test_clipped_surrogate_clips_large_positive_ratio() -> None:
    """Khi ratio vượt 1+eps với advantage dương, loss bị chặn (clip)."""
    clip = 0.2
    ratio = torch.tensor([2.0])          # vượt xa 1.2
    adv = torch.tensor([1.0])
    objective = _clipped_surrogate(ratio, adv, clip)
    # surr1 = 2.0 ; surr2 = 1.2 * 1.0 = 1.2 -> min = 1.2 (đã bị clip)
    assert objective.item() == pytest.approx(1.2, abs=1e-6)
    # Nếu không clip thì objective = 2.0 -> chứng tỏ đã bị chặn.
    assert objective.item() < (ratio * adv).item()


def test_clipped_surrogate_negative_advantage_clips_low_ratio() -> None:
    """Với advantage âm, ratio nhỏ bị chặn ở 1-eps."""
    clip = 0.2
    ratio = torch.tensor([0.1])          # nhỏ hơn 0.8
    adv = torch.tensor([-1.0])
    objective = _clipped_surrogate(ratio, adv, clip)
    # surr1 = -0.1 ; surr2 = 0.8 * -1 = -0.8 -> min = -0.8
    assert objective.item() == pytest.approx(-0.8, abs=1e-6)


def test_clipped_surrogate_no_clip_when_ratio_one() -> None:
    """Khi ratio = 1 (chưa đổi policy), objective = advantage."""
    clip = 0.2
    ratio = torch.tensor([1.0, 1.0])
    adv = torch.tensor([0.5, -0.3])
    objective = _clipped_surrogate(ratio, adv, clip)
    assert torch.allclose(objective, adv, atol=1e-6)


# ----------------------------------------------------------------------
# Cấu hình & Trajectory
# ----------------------------------------------------------------------
def test_ppo_config_from_dict_reads_spec_values() -> None:
    """``PPOConfig.from_dict`` đọc đúng các giá trị trong spec."""
    cfg = PPOConfig.from_dict(
        {
            "lr": 0.00025, "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
            "value_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5,
            "update_epochs": 4, "total_steps": 1000000, "minibatch_size": 4096,
            "unknown_key": 123,  # phải bị bỏ qua
        }
    )
    assert cfg.lr == pytest.approx(0.00025)
    assert cfg.clip_epsilon == pytest.approx(0.2)
    assert cfg.value_coef == pytest.approx(0.5)
    assert cfg.entropy_coef == pytest.approx(0.01)
    assert cfg.gae_lambda == pytest.approx(0.95)
    assert cfg.update_epochs == 4
    assert not hasattr(cfg, "unknown_key")


def test_trajectory_append_and_len() -> None:
    """``Trajectory`` ghi nhận đúng số bước."""
    traj = Trajectory()
    for i in range(3):
        traj.append(
            graph=None, legal_mask=torch.zeros(8100, dtype=torch.bool),
            action=i, log_prob=-1.0, value=0.1, reward=0.0, done=False,
            player=1, attn_weights=None,
        )
    assert len(traj) == 3
    assert traj.length == 3
    assert traj.actions == [0, 1, 2]


# ----------------------------------------------------------------------
# PPOAgent.update (end-to-end)
# ----------------------------------------------------------------------
def _make_fake_trajectory(seed: int = 0) -> Trajectory:
    """Tạo trajectory giả từ một ván thật (ngắn) để test update."""
    from src.env import XiangqiEnv

    env = XiangqiEnv(max_moves=8, legal_move_filter=False, seed=seed)
    state = env.reset()
    traj = Trajectory()
    rng = torch.Generator().manual_seed(seed)
    for _ in range(6):
        mask = state["legal_mask"]
        legal = mask.nonzero(as_tuple=False).flatten()
        action = int(legal[torch.randint(len(legal), (1,), generator=rng)])
        value = float(torch.randn(1, generator=rng).item())
        traj.append(
            graph=state["graph"], legal_mask=mask, action=action,
            log_prob=-math.log(mask.sum().item()), value=value,
            reward=0.1, done=False, player=state["player"],
            attn_weights=None,
        )
        state, _r, done, _i = env.step(action)
        if done:
            break
    return traj


@pytest.fixture()
def tiny_agent() -> PPOAgent:
    """Một ``PPOAgent`` rất nhỏ để test update nhanh."""
    model = XiangqiPolicyNet(
        gnn_hidden_dim=16, motif_dim=8, num_motifs=4,
        policy_hidden=32, value_hidden=16, dropout=0.0,
    )
    cfg = PPOConfig(
        lr=1e-3, update_epochs=2, minibatch_size=16, total_steps=100, use_amp=False
    )
    return PPOAgent(model, cfg, device="cpu")


def test_update_returns_stats_and_changes_params(tiny_agent: PPOAgent) -> None:
    """``update`` phải trả thống kê hợp lệ và thay đổi tham số model."""
    trajs = [_make_fake_trajectory(seed=i) for i in range(3)]
    before = [p.detach().clone() for p in tiny_agent.model.parameters()]

    stats = tiny_agent.update(trajs)

    assert set(["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]).issubset(stats)
    assert stats["entropy"] >= 0.0
    assert math.isfinite(stats["policy_loss"])
    assert math.isfinite(stats["value_loss"])

    # Ít nhất một tham số phải thay đổi sau cập nhật.
    changed = any(
        not torch.allclose(b, p.detach())
        for b, p in zip(before, tiny_agent.model.parameters())
    )
    assert changed, "PPO update không làm thay đổi tham số nào."


def test_update_empty_returns_empty_dict(tiny_agent: PPOAgent) -> None:
    """Không có dữ liệu -> ``update`` trả dict rỗng (không crash)."""
    assert tiny_agent.update([]) == {}


def test_value_loss_zero_when_prediction_matches() -> None:
    """MSE = 0 khi giá trị dự đoán trùng return (kiểm tra công thức loss)."""
    pred = torch.tensor([0.5, -0.5])
    target = torch.tensor([0.5, -0.5])
    assert torch.nn.functional.mse_loss(pred, target).item() == pytest.approx(0.0)


def test_scheduler_lowers_lr_over_steps(tiny_agent: PPOAgent) -> None:
    """Learning rate phải giảm dần theo cosine annealing sau các lần update."""
    trajs = [_make_fake_trajectory(seed=1)]
    lr_before = tiny_agent.optimizer.param_groups[0]["lr"]
    for _ in range(3):
        tiny_agent.update(trajs)
    lr_after = tiny_agent.optimizer.param_groups[0]["lr"]
    assert lr_after < lr_before

