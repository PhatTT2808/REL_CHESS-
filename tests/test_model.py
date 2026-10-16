"""Kiểm thử tầng model: forward shape, legal mask, gradient, act/evaluate."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.env import XiangqiEnv
from src.env.constants import ACTION_SPACE
from src.models import GNNEncoder, MotifAttention, XiangqiPolicyNet


@pytest.fixture()
def batched_graphs():
    """Batch gồm 4 thế cờ liên tiếp từ một ván."""
    env = XiangqiEnv(max_moves=200, seed=5)
    state = env.reset()
    graphs, masks = [], []
    for _ in range(4):
        graphs.append(state["graph"])
        masks.append(state["legal_mask"])
        action = int(state["legal_mask"].nonzero()[0])
        state, _r, done, _i = env.step(action)
        if done:
            break
    batch = Batch.from_data_list(graphs)
    mask = torch.stack(masks[: batch.num_graphs])
    return batch, mask


def test_gnn_encoder_output_shapes(batched_graphs) -> None:
    """``GNNEncoder`` phải trả graph_emb (B, hidden) và node_emb (N, hidden)."""
    batch, _mask = batched_graphs
    encoder = GNNEncoder(hidden_dim=32, num_layers=3, heads=4, dropout=0.0)
    graph_emb, node_emb = encoder(batch)
    assert graph_emb.shape == (batch.num_graphs, 32)
    assert node_emb.shape == (batch.num_nodes, 32)


def test_motif_attention_output_shapes(batched_graphs) -> None:
    """``MotifAttention`` phải trả context (B, motif_dim) và attn (B, num_motifs)."""
    batch, _mask = batched_graphs
    attn_module = MotifAttention(graph_dim=32, motif_dim=16, num_motifs=8)
    graph_emb = torch.randn(batch.num_graphs, 32)
    context, weights = attn_module(graph_emb)
    assert context.shape == (batch.num_graphs, 16)
    assert weights.shape == (batch.num_graphs, 8)
    # Attention weights phải không âm và tổng ~1 theo num_motifs.
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch.num_graphs), atol=1e-4)
    # Motif bank phải là tham số học được với shape đúng.
    assert attn_module.motif_bank.shape == (8, 16)


def test_policy_forward_shapes(batched_graphs, small_model) -> None:
    """Forward trả logits (B, 8100), value (B,), attn (B, num_motifs)."""
    batch, mask = batched_graphs
    out = small_model(batch, mask)
    b = batch.num_graphs
    assert out.logits.shape == (b, ACTION_SPACE)
    assert out.value.shape == (b,)
    assert torch.isfinite(out.value).all()
    assert out.attn_weights.shape == (b, small_model.num_motifs)
    assert out.graph_emb.shape == (b, small_model.gnn_hidden_dim)
    # Value KHÔNG bị tanh kẹp nữa: thang giá trị tự do để biểu diễn return
    # (terminal ±1 + shaping nhẹ). Trường hợp khởi tạo có thể vẫn nằm [-1,1]
    # nên chỉ kiểm tra finite.


def test_shared_dim_is_192() -> None:
    """Kích thước đặc trưng chia sẻ phải = gnn_hidden_dim + motif_dim (=192)."""
    model = XiangqiPolicyNet(gnn_hidden_dim=128, motif_dim=64)
    assert model.shared_dim == 192


def test_legal_mask_zeroes_probabilities(batched_graphs, small_model) -> None:
    """Xác suất của các hành động ngoài legal mask phải bằng 0."""
    batch, mask = batched_graphs
    out = small_model(batch, mask)
    probs = F.softmax(out.logits, dim=-1)
    assert probs[~mask].max().item() == pytest.approx(0.0, abs=1e-9)
    # Trong mask phải có xác suất dương.
    assert probs[mask].min().item() > 0.0


def test_logits_invalid_actions_are_neg_inf(batched_graphs, small_model) -> None:
    """Logits của hành động không hợp lệ phải được đặt về ``NEG_INF``."""
    from src.models.policy_net import NEG_INF

    batch, mask = batched_graphs
    out = small_model(batch, mask)
    assert out.logits[~mask].max().item() == pytest.approx(NEG_INF, rel=1e-6)


def test_act_returns_legal_actions(batched_graphs, small_model) -> None:
    """``act`` phải chỉ trả về hành động nằm trong legal mask."""
    batch, mask = batched_graphs
    actions, log_probs, values, attn = small_model.act(batch, mask)
    assert actions.shape == (batch.num_graphs,)
    assert log_probs.shape == (batch.num_graphs,)
    assert values.shape == (batch.num_graphs,)
    # Mọi hành động chọn ra đều hợp lệ.
    assert bool(mask.gather(1, actions.unsqueeze(1)).all())
    # log-prob phải <= 0.
    assert (log_probs <= 0).all()


def test_act_deterministic_is_argmax(batched_graphs, small_model) -> None:
    """Chế độ deterministic phải chọn argmax của logits."""
    batch, mask = batched_graphs
    actions, _, _, _ = small_model.act(batch, mask, deterministic=True)
    out = small_model(batch, mask)
    assert torch.equal(actions, out.logits.argmax(dim=-1))


def test_evaluate_shapes_and_entropy(batched_graphs, small_model) -> None:
    """``evaluate`` trả đúng shape; entropy phải >= 0."""
    batch, mask = batched_graphs
    actions = torch.randint(0, ACTION_SPACE, (batch.num_graphs,))
    # Đảm bảo hành động hợp lệ để log_prob hữu hạn.
    actions = torch.tensor(
        [int(m.nonzero()[0]) for m in mask], dtype=torch.long
    )
    log_probs, entropy, values = small_model.evaluate(batch, actions, mask)
    assert log_probs.shape == (batch.num_graphs,)
    assert entropy.shape == (batch.num_graphs,)
    assert values.shape == (batch.num_graphs,)
    assert (entropy >= 0).all()


def test_iccs_to_action_roundtrip() -> None:
    """Mã hoá ICCS -> action phải đúng ánh xạ hành động của env.

    Quy ước ĐÃ xác minh trên 2000 mẫu dataset: ``y_cchess == rank``,
    100% nước ``played`` hợp lệ theo ``cchess.ChessBoard``.
    """
    from src.env.constants import pos_to_index
    from src.training.bc_dataset import iccs_to_action

    # g2e3: từ (6, 2) tới (4, 3) — giống ví dụ trong XiangqiEnv.
    a = iccs_to_action("g2e3")
    assert a == pos_to_index(6, 2) * 90 + pos_to_index(4, 3)

    # Mã sai phải raise thay vì trả số rác.
    with pytest.raises(ValueError):
        iccs_to_action("z9z9")


def test_board90_to_grid_length() -> None:
    """board90 ngắn/dài bất thường phải raise ngay."""
    from src.training.bc_dataset import board90_to_grid

    with pytest.raises(ValueError):
        board90_to_grid("...")
    grid = board90_to_grid("." * 90)
    assert len(grid) == 10 and len(grid[0]) == 9


def test_evaluate_entropy_is_finite_and_masked(batched_graphs, small_model) -> None:
    """HỒI QUY cho bug vòng train 1e6 bước: entropy phải finite, không NaN.

    Nguyên nhân cũ: ``-(probs * log_probs_all).sum(-1)`` tính cả trên hành
    động ngoài legal mask, trong đó ``probs=0`` và ``log_probs=-inf`` nên
    ``0 * -inf = NaN``. NaN lan sang ``loss`` -> ``GradScaler`` bỏ qua
    ``optimizer.step()`` -> cả vòng train 1e6 bước không học gì.
    """
    batch, mask = batched_graphs
    actions = torch.tensor(
        [int(m.nonzero()[0]) for m in mask], dtype=torch.long
    )
    log_probs, entropy, _values = small_model.evaluate(batch, actions, mask)

    assert torch.isfinite(entropy).all(), (
        f"entropy chứa NaN/Inf: {entropy}. Bug 0 * -inf ngoài legal mask."
    )
    assert torch.isfinite(log_probs).all()
    assert (entropy >= 0).all()

    # NEG_INF phải biểu diễn được trong float16 (max 65504). Nếu vượt quá,
    # AMP sẽ đổi thành -inf -> grad inf -> GradScaler skip cả lần update.
    from src.models.policy_net import NEG_INF as _neg

    assert float(_neg) >= -65504.0, f"NEG_INF={_neg} tràn float16 thành -inf."

    # Entropy phải khớp công thức thủ công chỉ trên hành động hợp lệ.
    import torch.nn.functional as F

    with torch.no_grad():
        out = small_model(batch, mask)
        logp = F.log_softmax(out.logits, dim=-1)
        probs = logp.exp()
        ref = -(
            torch.where(mask, probs * logp, torch.zeros_like(probs)).sum(dim=-1)
        )
    assert torch.allclose(entropy, ref, atol=1e-5)


def test_backward_gradients_flow_everywhere(batched_graphs, small_model) -> None:
    """Backward phải tạo gradient cho TẤT CẢ tham số (kể cả motif_bank)."""
    batch, mask = batched_graphs
    actions = torch.tensor([int(m.nonzero()[0]) for m in mask], dtype=torch.long)

    log_probs, entropy, values = small_model.evaluate(batch, actions, mask)
    loss = -log_probs.mean() + (values ** 2).mean() - 0.01 * entropy.mean()
    loss.backward()

    missing = [n for n, p in small_model.named_parameters() if p.grad is None]
    assert missing == [], f"Các tham số không có gradient: {missing}"

    # Motif bank phải nhận gradient (đây là phần "symbolic" quan trọng).
    motif_grad = small_model.motif_attention.motif_bank.grad
    assert motif_grad is not None
    assert torch.isfinite(motif_grad).all()
    assert motif_grad.norm().item() > 0.0


def test_head_dimensions_model_defaults() -> None:
    """Kiểm tra dim của các head khớp công bố trong thiết kế (128/512/256)."""
    model = XiangqiPolicyNet(
        gnn_hidden_dim=128, motif_dim=64, policy_hidden=512, value_hidden=256
    )
    first_policy = model.policy_head[0]
    assert isinstance(first_policy, torch.nn.Linear)
    assert first_policy.in_features == 192
    assert first_policy.out_features == 512
    last_policy = model.policy_head[-1]
    assert last_policy.out_features == ACTION_SPACE
    assert model.value_head[0].in_features == 192
    assert model.value_head[-1].out_features == 1  # [-1] là Linear cuối (Linear 256->1)
    assert isinstance(model.value_head[-1], torch.nn.Linear)


def test_deterministic_forward_with_same_seed() -> None:
    """Cùng seed -> forward cho kết quả giống nhau."""
    from src.utils.seed import set_seed

    def run() -> torch.Tensor:
        set_seed(123)
        model = XiangqiPolicyNet(
            gnn_hidden_dim=32, motif_dim=16, num_motifs=8,
            policy_hidden=64, value_hidden=32, dropout=0.0,
        )
        env = XiangqiEnv(max_moves=200, seed=1)
        state = env.reset()
        batch = Batch.from_data_list([state["graph"]])
        out = model(batch, state["legal_mask"].unsqueeze(0))
        return out.logits

    assert torch.allclose(run(), run(), atol=1e-6)

