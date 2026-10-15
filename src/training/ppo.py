"""Thuật toán PPO (Proximal Policy Optimization) cho cờ tướng.

Bao gồm:
    - **GAE** (Generalized Advantage Estimation, lambda=0.95).
    - **Clipped surrogate objective** (clip=0.2).
    - **Value loss** (coef=0.5) + **entropy bonus** (coef=0.01).
    - **Gradient clipping** (max_norm=0.5), **Adam** + **cosine annealing**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Siêu tham số PPO."""

    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 512
    minibatch_size: int = 4096
    total_steps: int = 1_000_000
    use_amp: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "PPOConfig":
        """Khởi tạo từ mục ``training`` trong YAML (bỏ qua key không thuộc PPO)."""
        cfg = cfg or {}
        kwargs = {k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg}
        return cls(**kwargs)


@dataclass
class Trajectory:
    """Một trajectory của một ván cờ (hoặc một phần ván).

    Lưu đủ dữ liệu để tính GAE và cập nhật PPO. Tensor giữ trên CPU để tiết
    kiệm VRAM; chỉ chuyển sang GPU khi học.
    """

    graphs: List[Any] = field(default_factory=list)                # list[PyG Data]
    legal_masks: List[torch.Tensor] = field(default_factory=list)  # (8100,) bool
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    players: List[int] = field(default_factory=list)               # màu bên đi mỗi nước
    attn_weights: List[torch.Tensor] = field(default_factory=list)
    result: Optional[str] = None
    reason: str = "ongoing"
    length: int = 0

    def append(
        self,
        graph,
        legal_mask: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        player: int,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Thêm một bước vào trajectory."""
        self.graphs.append(graph)
        self.legal_masks.append(legal_mask)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.players.append(int(player))
        if attn_weights is not None:
            self.attn_weights.append(attn_weights)
        self.length += 1

    def __len__(self) -> int:
        return len(self.actions)


def compute_gae(
    rewards: List[float] | torch.Tensor,
    values: List[float] | torch.Tensor,
    dones: List[bool] | torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tính advantage và return bằng Generalized Advantage Estimation.

    Công thức::

        delta_t  = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t      = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        return_t = A_t + V(s_t)

    Args:
        rewards: Phần thưởng từng bước.
        values: Giá trị V(s_t) từng bước.
        dones: Cờ kết thúc episode từng bước.
        gamma: Hệ số chiết khấu.
        gae_lambda: Hệ số lambda của GAE.
        next_value: V(s_{T+1}) để bootstrap (thường = 0 vì ván đã kết thúc).

    Returns:
        ``(advantages, returns)`` — hai tensor cùng độ dài với ``rewards``.
    """
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    values_t = torch.as_tensor(values, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, dtype=torch.float32)
    n = rewards_t.shape[0]

    advantages = torch.zeros(n, dtype=torch.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_val = next_value
            next_non_terminal = 1.0
        else:
            next_val = values_t[t + 1]
            next_non_terminal = 1.0 - dones_t[t + 1]
        delta = rewards_t[t] + gamma * next_val * next_non_terminal - values_t[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_t
    return advantages, returns


class PPOAgent:
    """Bọc mô hình + optimizer + vòng lặp cập nhật PPO.

    Args:
        model: ``XiangqiPolicyNet``.
        config: ``PPOConfig``.
        device: Thiết bị torch (CPU hoặc CUDA).
    """

    def __init__(
        self,
        model: nn.Module,
        config: PPOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # Cosine annealing: giảm dần lr qua toàn bộ ``total_steps``.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, config.total_steps), eta_min=config.lr * 0.05
        )
        self.use_amp = bool(config.use_amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # --- Chống "train chết âm thầm" ---
        # ``_init_scale``: scale ban đầu của GradScaler. Khi GradScaler phát hiện
        # inf/NaN trong grad, nó GIẢM scale và bỏ qua ``optimizer.step()``. Ta so
        # scale hiện tại với scale ban đầu để phát hiện việc skip này.
        self._init_scale = float(self.scaler.get_scale())
        # Tổng số minibatch đã bị skip kể từ lúc khởi tạo agent.
        self.skipped_steps = 0

        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Thu thập dữ liệu (dùng khi sinh self-play)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_step(
        self, graph, legal_mask: torch.Tensor
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Chọn một hành động cho MỘT thế cờ.

        Args:
            graph: ``Data`` của một thế cờ (chưa batch).
            legal_mask: ``(8100,)`` bool.

        Returns:
            ``(action, log_prob, value, attn_weights)``.
        """
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([graph]).to(self.device)
        mask = legal_mask.unsqueeze(0).to(self.device)
        actions, log_probs, values, attn = self.model.act(
            batch, mask, deterministic=False
        )
        return (
            int(actions.item()),
            float(log_probs.item()),
            float(values.item()),
            attn.squeeze(0).cpu(),
        )

    @torch.no_grad()
    def act_deterministic(self, graph, legal_mask: torch.Tensor) -> int:
        """Chọn hành động tốt nhất (argmax) — dùng khi đánh giá."""
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([graph]).to(self.device)
        mask = legal_mask.unsqueeze(0).to(self.device)
        actions, _, _, _ = self.model.act(batch, mask, deterministic=True)
        return int(actions.item())

    # ------------------------------------------------------------------
    # Cập nhật PPO
    # ------------------------------------------------------------------
    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Cập nhật mô hình từ danh sách trajectory.

        Quy trình:
            1. Tính GAE cho từng ván riêng (không trộn giữa các ván).
            2. Chuẩn hoá advantage (mean 0, std 1).
            3. Chia mini-batch, chạy ``update_epochs`` vòng SGD.

        Args:
            trajectories: Danh sách ``Trajectory`` từ self-play.

        Returns:
            Dict thống kê (policy_loss, value_loss, entropy, approx_kl...).
        """
        from torch_geometric.data import Batch

        all_graphs: List[Any] = []
        all_masks: List[torch.Tensor] = []
        all_actions: List[int] = []
        all_log_probs: List[float] = []
        all_advantages: List[torch.Tensor] = []
        all_returns: List[torch.Tensor] = []

        g, lam = self.config.gamma, self.config.gae_lambda
        for traj in trajectories:
            if len(traj) == 0:
                continue
            adv, ret = compute_gae(
                traj.rewards, traj.values, traj.dones, gamma=g, gae_lambda=lam
            )
            all_graphs.extend(traj.graphs)
            all_masks.extend(traj.legal_masks)
            all_actions.extend(traj.actions)
            all_log_probs.extend(traj.log_probs)
            all_advantages.append(adv)
            all_returns.append(ret)

        if not all_graphs:
            logger.warning("Không có dữ liệu để cập nhật PPO.")
            return {}

        advantages = torch.cat(all_advantages)
        returns_t = torch.cat(all_returns).to(dtype=torch.float32)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.as_tensor(all_log_probs, dtype=torch.float32)
        actions_t = torch.as_tensor(all_actions, dtype=torch.long)

        n = len(all_graphs)
        mb_size = max(1, min(self.config.minibatch_size, n))
        stats: Dict[str, float] = {}
        skipped_before = self.skipped_steps
        n_minibatches = 0

        for _epoch in range(self.config.update_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, mb_size):
                mb_idx = perm[start : start + mb_size]
                idx_list = mb_idx.tolist()
                n_minibatches += 1

                batch = Batch.from_data_list(
                    [all_graphs[i] for i in idx_list]
                ).to(self.device)
                mask = torch.stack([all_masks[i] for i in idx_list]).to(self.device)
                mb_actions = actions_t[mb_idx].to(self.device)
                mb_old_log_probs = old_log_probs[mb_idx].to(self.device)
                mb_adv = advantages[mb_idx].to(self.device)
                mb_returns = returns_t[mb_idx].to(self.device)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    log_probs, entropy, values = self.model.evaluate(
                        batch, mb_actions, mask
                    )
                    ratio = torch.exp(log_probs - mb_old_log_probs)

                    surr1 = ratio * mb_adv
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.clip_epsilon,
                            1.0 + self.config.clip_epsilon,
                        )
                        * mb_adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, mb_returns)
                    entropy_loss = -entropy.mean()
                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        + self.config.entropy_coef * entropy_loss
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "loss = NaN/Inf trong PPO update. "
                        "Thường do entropy tính cả trên hành động ngoài legal mask "
                        "(0 * -inf) hoặc logits/log_prob không hữu hạn. "
                        "Kiểm tra lại XiangqiPolicyNet.evaluate()."
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # GradScaler bỏ qua step khi gặp inf/NaN trong grad. Ghi lại
                # số lần bị skip để phát hiện "train chết âm thầm" (chính là
                # chuyện đã xảy ra ở vòng train 1e6 bước trước).
                was_skipped = bool(
                    self.scaler.get_scale() < self._init_scale
                )
                if was_skipped:
                    self.skipped_steps += 1
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (log_probs - mb_old_log_probs)).mean()
                    clip_frac = (
                        (ratio - 1.0).abs() > self.config.clip_epsilon
                    ).float().mean()
                stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_frac": float(clip_frac.item()),
                    "grad_norm": float(grad_norm),
                    "loss": float(loss.item()),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }

        self.scheduler.step()
        self.last_stats = stats

        stats["skipped_steps"] = self.skipped_steps
        stats["minibatches"] = n_minibatches

        # Cảnh báo nếu cả lần update đều bị GradScaler bỏ qua -> mô hình
        # không học được gì dù vẫn "chạy" suốt 1e6 bước.
        skipped_now = self.skipped_steps - skipped_before
        if skipped_now > 0:
            logger.warning(
                "GradScaler đã bỏ qua %d/%d minibatch trong lần update (grad inf/NaN).",
                skipped_now, n_minibatches,
            )
        return stats

