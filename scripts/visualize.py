#!/usr/bin/env python
"""Trực quan hóa kết quả huấn luyện.

Sinh các hình:
    1. **Reward/metrics curve**: policy loss, value loss, entropy, win-rate,
       số nước trung bình theo step.
    2. **Elo curve**: Elo ước lượng theo từng checkpoint (nếu có file
       ``elo_history.jsonl`` trong ``paths.logs``).
    3. **Attention heatmap**: trục x = nước đi, trục y = motif; cho thấy
       motif chiến thuật nào đang kích hoạt ở mỗi nước (1 ván mẫu).
    4. **Motif usage bar**: mức kích hoạt trung bình của từng motif.

Ví dụ:
    python scripts/visualize.py --config configs/smoke.yaml \
        --checkpoint checkpoints_smoke/step_2252.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.policy_net import XiangqiPolicyNet
from src.utils.checkpoint import load_checkpoint, load_metrics, list_checkpoints
from src.utils.config import ensure_dir, get_device, load_config, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger("visualize")

import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình (chạy trên server)
import matplotlib.pyplot as plt  # noqa: E402


def build_model_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Trích tham số kiến trúc model từ config."""
    m = config["model"]
    return {
        "gnn_hidden_dim": m["gnn_hidden_dim"],
        "gnn_layers": m["gnn_layers"],
        "gnn_heads": m.get("gnn_heads", 4),
        "motif_dim": m["motif_dim"],
        "num_motifs": m["num_motifs"],
        "policy_hidden": m["policy_hidden"],
        "value_hidden": m["value_hidden"],
        "dropout": m.get("dropout", 0.1),
    }


def plot_metrics_curves(records: List[Dict[str, Any]], figures_dir: Path) -> Optional[Path]:
    """Vẽ các đường cong metrics (loss, entropy, win-rate, avg_moves).

    Args:
        records: Danh sách bản ghi metrics đọc từ JSONL.
        figures_dir: Thư mục lưu hình.

    Returns:
        Đường dẫn file hình, hoặc ``None`` nếu không có dữ liệu.
    """
    if not records:
        logger.warning("Không có metrics để vẽ (chạy train_rl.py trước).")
        return None

    steps = [r["step"] for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        (axes[0, 0], "policy_loss", "Policy Loss", "tab:red"),
        (axes[0, 1], "value_loss", "Value Loss", "tab:blue"),
        (axes[1, 0], "entropy", "Entropy (khám phá)", "tab:green"),
        (axes[1, 1], "win_rate", "Win-rate (vs Opponent Pool)", "tab:purple"),
    ]
    for ax, key, title, color in panels:
        values = [r.get(key, float("nan")) for r in records]
        if all(v != v for v in values):  # tất cả NaN
            ax.text(0.5, 0.5, f"(không có dữ liệu: {key})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")
        ax.plot(steps, values, color=color, linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    fig.suptitle("Đường cong huấn luyện PPO Self-Play (Neural-Symbolic Xiangqi)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = figures_dir / "metrics_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Đã lưu hình metrics: %s", out)
    return out


def plot_elo_curve(config: Dict[str, Any], figures_dir: Path) -> Optional[Path]:
    """Vẽ đường Elo theo checkpoint, nếu có ``elo_history.jsonl``.

    File ``elo_history.jsonl`` (nếu tồn tại) có mỗi dòng dạng
    ``{"step": int, "elo": float, "ci_low": float, "ci_high": float}``.
    """
    import json

    path = resolve_path(config["paths"]["logs"]) / "elo_history.jsonl"
    if not path.is_file():
        logger.warning("Không có %s -> bỏ qua Elo curve.", path)
        return None

    steps, elos, lows, highs = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(rec["step"])
            elos.append(rec["elo"])
            lows.append(rec.get("ci_low", rec["elo"]))
            highs.append(rec.get("ci_high", rec["elo"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, elos, marker="o", color="tab:blue", label="Elo")
    ax.fill_between(steps, lows, highs, alpha=0.2, color="tab:blue",
                    label="CI 95%")
    ax.set_xlabel("step")
    ax.set_ylabel("Elo")
    ax.set_title("Đường Elo theo checkpoint")
    ax.legend()
    ax.grid(alpha=0.3)
    out = figures_dir / "elo_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Đã lưu hình Elo: %s", out)
    return out


def collect_attention_sequence(
    model: XiangqiPolicyNet,
    device,
    max_moves: int = 40,
    legal_move_filter: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, List[int], np.ndarray]:
    """Chạy một ván mẫu và thu attention weights tại mỗi nước đi.

    Args:
        model: Mô hình đã huấn luyện.
        device: Thiết bị.
        max_moves: Số nước tối đa của ván mẫu.
        legal_move_filter: Lọc nước tự-chiếu-tướng.
        seed: Seed.

    Returns:
        ``(attn_matrix, actions, values)``:
            - ``attn_matrix``: ``(num_moves, num_motifs)``.
            - ``actions``: danh sách action index mỗi nước.
            - ``values``: giá trị V(s) mỗi nước.
    """
    import torch
    from torch_geometric.data import Batch

    from src.env.xiangqi_env import XiangqiEnv

    env = XiangqiEnv(max_moves=max_moves, legal_move_filter=legal_move_filter, seed=seed)
    state = env.reset()

    attn_rows: List[np.ndarray] = []
    actions: List[int] = []
    values: List[float] = []

    model.eval()
    with torch.no_grad():
        while True:
            batch = Batch.from_data_list([state["graph"]]).to(device)
            mask = state["legal_mask"].unsqueeze(0).to(device)
            action, _lp, value, attn = model.act(batch, mask, deterministic=True)
            attn_rows.append(attn.squeeze(0).cpu().numpy())
            actions.append(int(action.item()))
            values.append(float(value.item()))

            state, _r, done, _info = env.step(int(action.item()))
            if done or len(actions) >= max_moves:
                break

    attn_matrix = np.stack(attn_rows, axis=0) if attn_rows else np.zeros((0, 1))
    return attn_matrix, actions, np.asarray(values, dtype=np.float32)


def plot_attention_heatmap(
    model: XiangqiPolicyNet,
    device,
    figures_dir: Path,
    max_moves: int = 40,
    legal_move_filter: bool = True,
    seed: int = 0,
) -> Optional[Path]:
    """Vẽ heatmap attention (motif x nước đi) cho một ván mẫu.

    Args:
        model: Mô hình đã huấn luyện.
        device: Thiết bị.
        figures_dir: Thư mục lưu hình.
        max_moves: Số nước tối đa của ván mẫu.
        legal_move_filter: Lọc nước tự-chiếu-tướng.
        seed: Seed.

    Returns:
        Đường dẫn file hình, hoặc ``None`` nếu không thu được dữ liệu.
    """
    attn, actions, _values = collect_attention_sequence(
        model, device, max_moves=max_moves,
        legal_move_filter=legal_move_filter, seed=seed,
    )
    if attn.size == 0:
        logger.warning("Không thu được attention (ván quá ngắn).")
        return None

    fig, ax = plt.subplots(figsize=(max(8, attn.shape[0] * 0.4), 7))
    im = ax.imshow(attn.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Nước đi (thứ tự)")
    ax.set_ylabel("Motif index")
    ax.set_title("Attention Heatmap: motif nào kích hoạt ở mỗi nước đi")
    fig.colorbar(im, ax=ax, label="Attention weight")
    out = figures_dir / "attention_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Đã lưu hình attention heatmap: %s", out)
    return out


def plot_motif_usage(
    model: XiangqiPolicyNet, device, figures_dir: Path, seed: int = 0
) -> Optional[Path]:
    """Vẽ biểu đồ mức sử dụng trung bình của từng motif (trên 1 ván mẫu)."""
    attn, _actions, _values = collect_attention_sequence(
        model, device, max_moves=40, seed=seed
    )
    if attn.size == 0:
        return None

    usage = attn.mean(axis=0)
    num_motifs = usage.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, num_motifs * 0.35), 4.5))
    ax.bar(np.arange(num_motifs), usage, color="tab:orange")
    ax.set_xlabel("Motif index")
    ax.set_ylabel("Mức kích hoạt trung bình")
    ax.set_title("Mức sử dụng motif trung bình trong ván mẫu")
    ax.grid(alpha=0.3, axis="y")
    out = figures_dir / "motif_usage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Đã lưu hình motif usage: %s", out)
    return out


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Trực quan hóa huấn luyện cờ tướng")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Đường dẫn file cấu hình YAML.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint để vẽ attention (mặc định: mới nhất trong paths.checkpoints).",
    )
    parser.add_argument(
        "--max-moves", type=int, default=40,
        help="Số nước của ván mẫu dùng cho attention heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    """Điểm vào của script trực quan hóa."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    logger.handlers = logging.getLogger().handlers

    figures_dir = ensure_dir(config["paths"]["figures"])
    logger.info("Thư mục hình: %s", figures_dir)

    # 1. Đường cong metrics.
    metrics_path = str(resolve_path(config["paths"]["logs"]) / "metrics.jsonl")
    records = load_metrics(metrics_path)
    plot_metrics_curves(records, figures_dir)

    # 2. Đường Elo (nếu có).
    plot_elo_curve(config, figures_dir)

    # 3-4. Attention heatmap + motif usage (cần checkpoint).
    device = get_device(config["training"]["device"])
    checkpoint = args.checkpoint
    if checkpoint is None:
        items = list_checkpoints(config["paths"]["checkpoints"])
        checkpoint = str(items[-1][1]) if items else None

    if checkpoint is None:
        logger.warning("Không có checkpoint -> bỏ qua attention heatmap.")
        return

    model = XiangqiPolicyNet(**build_model_kwargs(config))
    load_checkpoint(checkpoint, model, map_location=str(device))
    model.to(device).eval()

    plot_attention_heatmap(
        model, device, figures_dir,
        max_moves=args.max_moves,
        legal_move_filter=bool(config["env"]["legal_move_filter"]),
        seed=int(config.get("seed", 42)),
    )
    plot_motif_usage(model, device, figures_dir, seed=int(config.get("seed", 42)))
    logger.info("Trực quan hóa hoàn tất.")


if __name__ == "__main__":
    main()


