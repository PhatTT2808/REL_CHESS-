"""Mạng chính sách Neural-Symbolic cho cờ tướng.

Luồng dữ liệu (đúng thiết kế đồ án)::

    Graph ──► GNNEncoder ──► graph_emb (128)
                                 │
                                 └──► MotifAttention ──► tactical_context (64)
                                                          + attn_weights (30)
    Concat [graph_emb(128) | tactical_context(64)] = 192
        ├──► Policy Head ──► logits (8100)  + legal mask
        └──► Value Head  ──► value (1)

Ghi chú về Value head:
    Trước đây có ``nn.Tanh()`` — đã gỡ bỏ (vòng train 1e6 bước cho thấy tanh
    kẹp output trong [-1,1] nên không thể biễu diễn return thực, vì tổng reward
    một ván (potential-based + terminal) thường vượt ra ngoài. Sửa lại thành
    ``Linear`` thuần để thang giá trị không bị bóp méo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.env.constants import ACTION_SPACE
from src.models.gnn_encoder import GNNEncoder
from src.models.motif_attention import MotifAttention

logger = logging.getLogger(__name__)

# Giá trị dùng để "chặn" các hành động không hợp lệ trước softmax.
NEG_INF = -1e9


@dataclass
class PolicyOutput:
    """Kết quả forward của mạng chính sách."""

    logits: torch.Tensor  # (B, 8100) — đã áp legal mask
    value: torch.Tensor  # (B,)      — giá trị thế cờ, KHÔNG bị kẹp (Linear thuần)
    attn_weights: torch.Tensor  # (B, num_motifs)
    graph_emb: torch.Tensor  # (B, gnn_hidden_dim)


class XiangqiPolicyNet(nn.Module):
    """Mạng policy + value theo kiến trúc Neural-Symbolic.

    Args:
        gnn_hidden_dim: Số chiều ẩn của GNN.
        gnn_layers: Số lớp GAT.
        gnn_heads: Số attention head của GAT.
        motif_dim: Số chiều mỗi motif.
        num_motifs: Số motif trong bank.
        policy_hidden: Số chiều ẩn của policy head.
        value_hidden: Số chiều ẩn của value head.
        dropout: Tỷ lệ dropout dùng chung.
    """

    def __init__(
        self,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        motif_dim: int = 64,
        num_motifs: int = 30,
        policy_hidden: int = 512,
        value_hidden: int = 256,
        dropout: float = 0.1,
        action_space: int = ACTION_SPACE,
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.gnn_hidden_dim = gnn_hidden_dim
        self.motif_dim = motif_dim
        self.num_motifs = num_motifs

        self.encoder = GNNEncoder(
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=dropout,
        )
        self.motif_attention = MotifAttention(
            graph_dim=gnn_hidden_dim,
            motif_dim=motif_dim,
            num_motifs=num_motifs,
            num_heads=4,
            dropout=dropout,
        )

        # Kích thước đặc trưng chia sẻ: 128 + 64 = 192 (đúng thiết kế)
        self.shared_dim = gnn_hidden_dim + motif_dim

        self.policy_head = nn.Sequential(
            nn.Linear(self.shared_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, action_space),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.shared_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(
        self, data: Data, legal_mask: Optional[torch.Tensor] = None
    ) -> PolicyOutput:
        """Lan truyền xuôi toàn mạng.

        Args:
            data: Batch đồ thị (``Data`` hoặc ``Batch``).
            legal_mask: ``(B, 8100)`` bool — nếu có, logits của nước
                không hợp lệ bị đặt thành ``NEG_INF``.

        Returns:
            ``PolicyOutput`` chứa logits (đã mask), value, attn_weights.
        """
        graph_emb, _node_emb = self.encoder(data)
        tactical_ctx, attn_weights = self.motif_attention(graph_emb)

        shared = torch.cat([graph_emb, tactical_ctx], dim=-1)  # (B, 192)
        logits = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)

        logits = self.apply_legal_mask(logits, legal_mask)
        return PolicyOutput(
            logits=logits,
            value=value,
            attn_weights=attn_weights,
            graph_emb=graph_emb,
        )

    @staticmethod
    def apply_legal_mask(
        logits: torch.Tensor, legal_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Đặt logits của hành động không hợp lệ về ``NEG_INF``.

        Bảo đảm mỗi hàng còn ít nhất 1 hành động hợp lệ; nếu ``legal_mask``
        không có hành động nào (ván đã kết thúc), cho phép tất cả để tránh NaN.
        """
        if legal_mask is None:
            return logits
        mask = legal_mask.to(dtype=torch.bool, device=logits.device)
        empty_rows = ~mask.any(dim=-1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows] = True

        # Ép kiểu NEG_INF sang đúng dtype và device của logits để tránh lỗi overflow với float16
        neg_inf = torch.tensor(NEG_INF, dtype=logits.dtype, device=logits.device)
        return logits.masked_fill(~mask, neg_inf)

    # ------------------------------------------------------------------
    # Lấy hành động / đánh giá
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        data: Data,
        legal_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lấy mẫu hành động từ phân phối chính sách.

        Args:
            data: Batch đồ thị.
            legal_mask: ``(B, 8100)`` bool.
            deterministic: Nếu ``True``, chọn ``argmax`` thay vì lấy mẫu.

        Returns:
            ``(actions, log_probs, values, attn_weights)`` — tất cả theo batch.
        """
        out = self.forward(data, legal_mask)
        log_probs_all = F.log_softmax(out.logits, dim=-1)

        if deterministic:
            actions = torch.argmax(out.logits, dim=-1)
        else:
            probs = log_probs_all.exp()
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_probs = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, log_probs, out.value, out.attn_weights

    def evaluate(
        self,
        data: Data,
        actions: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Đánh giá lại ``(log_prob, entropy, value)`` cho hành động đã chọn.

        Dùng trong bước cập nhật PPO (tính lại log-prob sau khi weights đổi).

        Args:
            data: Batch đồ thị.
            actions: ``(B,)`` hành động đã lấy mẫu.
            legal_mask: ``(B, 8100)`` bool.

        Returns:
            ``(log_probs, entropy, values)``.
        """
        out = self.forward(data, legal_mask)
        log_probs_all = F.log_softmax(out.logits, dim=-1)
        probs = log_probs_all.exp()

        log_probs = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        # Entropy của phân phối đã mask (prob ngoài mask ~ 0 nên số hạng an toàn)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_probs, entropy, out.value

    @property
    def num_parameters(self) -> int:
        """Tổng số tham số có thể huấn luyện."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
