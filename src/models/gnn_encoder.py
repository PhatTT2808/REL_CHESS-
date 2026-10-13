"""Mã hóa thế cờ bằng GNN (Graph Attention Network).

Kiến trúc: 3 lớp ``GATConv`` với BatchNorm + ReLU + residual, sau đó gộp
node embedding thành **graph embedding** bằng ``global_mean_pool``.

Ý tưởng Neural-Symbolic ở đây: thay vì coi bàn cờ là ảnh, ta để GNN lan
truyền thông tin theo **quan hệ tấn công/phòng thủ/cùng đường** giữa các quân,
nhờ đó học được các pattern chiến thuật cục bộ (đôi xe, pháo đầu, mã đá...).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from src.env.constants import NODE_FEATURE_DIM, NUM_EDGE_TYPES


class GNNEncoder(nn.Module):
    """Encoder đồ thị thế cờ bằng 3 lớp GAT (Graph Attention).

    Args:
        hidden_dim: Số chiều ẩn của mỗi lớp GAT.
        num_layers: Số lớp GAT (mặc định 3 theo thiết kế).
        heads: Số attention head mỗi lớp (multi-head attention).
        dropout: Tỷ lệ dropout.
        in_dim: Số chiều feature đầu vào của node (mặc định 10).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        in_dim: int = NODE_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Lớp đầu: từ in_dim -> hidden_dim (giữ nguyên số chiều ra để cộng residual)
        self.convs.append(
            GATConv(in_dim, hidden_dim // heads, heads=heads, edge_dim=NUM_EDGE_TYPES)
        )
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=NUM_EDGE_TYPES,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lan truyền qua các lớp GAT.

        Args:
            data: ``Data`` với ``x``, ``edge_index``, ``edge_attr``, ``batch``.

        Returns:
            ``(graph_emb, node_emb)``:
            - ``graph_emb``: ``(B, hidden_dim)`` — embedding toàn cục.
            - ``node_emb``: ``(N, hidden_dim)`` — embedding từng quân (để visualize).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = x.new_zeros((edge_index.size(1), NUM_EDGE_TYPES))

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            identity = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            # Residual: chỉ cộng khi số chiều khớp (từ lớp thứ 2 trở đi)
            if identity.shape[-1] == x.shape[-1]:
                x = x + identity
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x
        graph_emb = self._pool(node_emb, data)
        return graph_emb, node_emb

    @staticmethod
    def _pool(node_emb: torch.Tensor, data: Data) -> torch.Tensor:
        """Gộp node embedding thành graph embedding (chịu được batch rỗng)."""
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        if node_emb.size(0) == 0:
            # Đồ thị rỗng: trả vector 0 với số graph suy ra từ ``batch``
            n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
            return node_emb.new_zeros((n_graphs, node_emb.size(1) if node_emb.dim() > 1 else 0))
        return global_mean_pool(node_emb, batch)
