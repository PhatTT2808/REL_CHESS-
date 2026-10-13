"""Tactical Motif Attention — cross-attention giữa thế cờ và "ngân hàng motif".

Ý tưởng Neural-Symbolic: con người chơi cờ tướng bằng các **mô-típ chiến thuật
có tên** (pháo đầu, bình phong mã, đôi xe, mã hậu pháo...). Ta mô hình hoá các
mô-típ này thành một **Motif Bank** gồm ``num_motifs`` vector học được
(``nn.Parameter``), rồi cho graph embedding **cross-attend** vào bank đó.

Kết quả:
    - ``tactical_context``: vector ngữ cảnh chiến thuật (``motif_dim``).
    - ``attn_weights``: trọng số chú ý lên từng motif (``num_motifs``) —
      dùng để visualize "motif nào đang kích hoạt ở mỗi nước đi".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MotifAttention(nn.Module):
    """Cross-attention: Query = graph embedding, Key/Value = Motif Bank.

    Args:
        graph_dim: Số chiều của graph embedding (đầu ra GNN).
        motif_dim: Số chiều của mỗi motif vector.
        num_motifs: Số lượng motif trong bank (mặc định 30).
        num_heads: Số attention head.
        dropout: Tỷ lệ dropout trong attention.
    """

    def __init__(
        self,
        graph_dim: int = 128,
        motif_dim: int = 64,
        num_motifs: int = 30,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        self.motif_dim = motif_dim
        self.num_motifs = num_motifs
        self.num_heads = num_heads

        # Motif Bank: tham số học được, mỗi hàng là 1 mô-típ chiến thuật.
        self.motif_bank = nn.Parameter(torch.randn(num_motifs, motif_dim) * 0.02)

        # Chiếu graph embedding -> không gian motif (đóng vai Query)
        self.query_proj = nn.Linear(graph_dim, motif_dim)
        # Chiếu motif bank -> Key/Value (dùng chung 1 phép chiếu tuyến tính)
        self.key_proj = nn.Linear(motif_dim, motif_dim)
        self.value_proj = nn.Linear(motif_dim, motif_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=motif_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(motif_dim)

    def forward(self, graph_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tính ngữ cảnh chiến thuật và trọng số attention.

        Args:
            graph_emb: ``(B, graph_dim)`` — embedding thế cờ từ GNN.

        Returns:
            ``(tactical_context, attn_weights)``:
            - ``tactical_context``: ``(B, motif_dim)``.
            - ``attn_weights``: ``(B, num_motifs)`` — trung bình qua các head,
              tổng theo chiều motif (mỗi phần tử là mức kích hoạt của 1 motif).
        """
        batch_size = graph_emb.size(0)

        # Query: (B, 1, motif_dim) — mỗi thế cờ là 1 truy vấn
        query = self.query_proj(graph_emb).unsqueeze(1)

        # Key/Value từ motif bank: (B, num_motifs, motif_dim)
        bank = self.motif_bank.unsqueeze(0).expand(batch_size, -1, -1)
        key = self.key_proj(bank)
        value = self.value_proj(bank)

        # Cross-attention (need_weights=True để lấy attention map cho visualize)
        context, weights = self.attn(
            query, key, value, need_weights=True, average_attn_weights=True
        )

        context = context.squeeze(1)           # (B, motif_dim)
        context = self.norm(context + self.query_proj(graph_emb))
        attn_weights = weights.squeeze(1)       # (B, num_motifs)
        return context, attn_weights
