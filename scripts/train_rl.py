#!/usr/bin/env python
"""Script huấn luyện chính: PPO self-play cho cờ tướng (Neural-Symbolic).

Vòng lặp:
    1. Sinh batch ván bằng self-play (agent đấu Opponent Pool theo PFSP).
    2. Tính GAE + cập nhật PPO (4 epochs).
    3. Ghi log metrics, lưu checkpoint định kỳ, snapshot vào pool.

Ví dụ:
    # Chạy nhanh trên CPU (khuyến nghị để kiểm thử)
    python scripts/train_rl.py --config configs/smoke.yaml

    # Chạy đầy đủ trên GPU Linux
    python scripts/train_rl.py --config configs/default.yaml

    # Tiếp tục từ checkpoint
    python scripts/train_rl.py --config configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Thêm thư mục gốc dự án vào sys.path để import ``src.*`` khi chạy script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.policy_net import XiangqiPolicyNet
from src.training.ppo import PPOAgent, PPOConfig
from src.training.reward import RewardConfig
from src.training.self_play import OpponentPool, SelfPlayRunner
from src.utils.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_metrics,
)
from src.utils.config import (
    ensure_dir,
    get_device,
    load_config,
    resolve_num_workers,
    resolve_path,
)
from src.utils.seed import set_seed

logger = logging.getLogger("train_rl")


def setup_logging(log_dir: str, level: int = logging.INFO) -> Path:
    """Cấu hình logging ra console + file.

    Args:
        log_dir: Thư mục chứa file log (từ config ``paths.logs``).
        level: Mức log.

    Returns:
        Đường dẫn file log.
    """
    directory = ensure_dir(log_dir)
    log_file = directory / "train.log"

    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return log_file


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


def train(config: Dict[str, Any], resume: bool = False) -> None:
    """Chạy vòng lặp huấn luyện PPO self-play.

    Args:
        config: Dict cấu hình đã nạp từ YAML.
        resume: Nếu ``True``, nạp checkpoint mới nhất để tiếp tục.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    paths = config["paths"]
    train_cfg = config["training"]
    env_cfg = config["env"]

    log_file = setup_logging(paths["logs"])
    device = get_device(train_cfg["device"])
    num_workers = resolve_num_workers(
        train_cfg.get("num_workers", 0), train_cfg["device"]
    )

    logger.info("=" * 70)
    logger.info("Bắt đầu huấn luyện Neural-Symbolic Xiangqi RL")
    logger.info("Config: %s", config.get("_config_path"))
    logger.info("Device: %s | num_workers (tham khảo): %s", device, num_workers)
    logger.info("Seed: %d | log: %s", seed, log_file)
    logger.info("=" * 70)

    # --- Khởi tạo model / agent / pool ---
    model_kwargs = build_model_kwargs(config)
    model = XiangqiPolicyNet(**model_kwargs)
    agent = PPOAgent(model, PPOConfig.from_dict(train_cfg), device=device)
    logger.info("Số tham số model: %s", f"{model.num_parameters:,}")

    reward_config = RewardConfig.from_dict(config.get("reward"))
    pool = OpponentPool(
        pool_dir=paths["pool"], model_kwargs=model_kwargs, device=device
    )

    start_step = 0
    if resume:
        ckpt = latest_checkpoint(paths["checkpoints"])
        if ckpt is not None:
            meta = load_checkpoint(
                ckpt, model, agent.optimizer, map_location=str(device)
            )
            start_step = int(meta.get("step", 0))
            logger.info("Tiếp tục huấn luyện từ step %d", start_step)
        else:
            logger.warning("Không tìm thấy checkpoint để resume; bắt đầu từ đầu.")
    if len(pool) == 0:
        pool.add_snapshot(model, 0)

    runner = SelfPlayRunner(
        agent=agent,
        pool=pool,
        reward_config=reward_config,
        max_moves=env_cfg["max_moves"],
        perpetual_check_limit=env_cfg["perpetual_check_limit"],
        legal_move_filter=env_cfg["legal_move_filter"],
        model_kwargs=model_kwargs,
        device=device,
        seed=seed,
    )

    games_per_batch = int(train_cfg["games_per_batch"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    snapshot_interval = int(train_cfg.get("pool_snapshot_interval", 5))
    total_steps = int(train_cfg["total_steps"])
    metrics_path = str(Path(paths["logs"]) / "metrics.jsonl")

    step = start_step
    batch_idx = 0
    last_checkpoint_step = step
    t_start = time.time()
    steps_since_start = 0

    while step < total_steps:
        batch_idx += 1
        t_batch = time.time()

        # 1. Sinh batch self-play.
        trajectories, stats = runner.generate_batch(games_per_batch)
        if not trajectories:
            logger.warning("Batch rỗng, bỏ qua batch này.")
            continue

        # 2. Cập nhật PPO.
        update_stats = agent.update(trajectories)

        step += stats.steps
        steps_since_start += stats.steps
        total_elapsed = time.time() - t_start
        sps = steps_since_start / max(total_elapsed, 1e-6)
        eta_seconds = (total_steps - step) / max(sps, 1e-6)

        # 3. Ghi log metrics.
        metrics = {
            "step": step,
            "batch": batch_idx,
            "games": stats.games,
            "steps": stats.steps,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "win_rate": stats.wins / max(stats.games, 1),
            "avg_moves": stats.avg_moves,
            "reasons": stats.reasons,
            "batch_seconds": round(time.time() - t_batch, 2),
            "steps_per_second": round(sps, 2),
            "eta_hours": round(eta_seconds / 3600.0, 3),
            **{k: round(v, 5) for k, v in update_stats.items()},
        }
        save_metrics(metrics, metrics_path)

        logger.info(
            "step %7d/%d | batch %4d | games %3d | T/B/H %2d/%2d/%2d | "
            "avg_moves %5.1f | pol %.4f val %.4f ent %.3f kl %.4f | "
            "%.1f st/s | ETA %.2fh",
            step, total_steps, batch_idx, stats.games,
            stats.wins, stats.losses, stats.draws, stats.avg_moves,
            update_stats.get("policy_loss", 0.0),
            update_stats.get("value_loss", 0.0),
            update_stats.get("entropy", 0.0),
            update_stats.get("approx_kl", 0.0),
            sps, eta_seconds / 3600.0,
        )

        # 4. Snapshot vào pool định kỳ (đối thủ mới cho PFSP).
        if batch_idx % snapshot_interval == 0:
            pool.add_snapshot(model, step)

        # 5. Lưu checkpoint định kỳ.
        if step - last_checkpoint_step >= checkpoint_interval:
            save_checkpoint(
                model, agent.optimizer, step, paths["checkpoints"],
                extra={"batch": batch_idx, "metrics": metrics},
            )
            last_checkpoint_step = step

    # Lưu checkpoint cuối cùng.
    save_checkpoint(
        model, agent.optimizer, step, paths["checkpoints"],
        extra={"batch": batch_idx, "final": True},
    )
    logger.info(
        "Hoàn tất huấn luyện: %d bước trong %.2f giờ.",
        step, (time.time() - t_start) / 3600.0,
    )


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Huấn luyện PPO self-play cờ tướng")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Đường dẫn file cấu hình YAML (tương đối từ gốc dự án).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Tiếp tục từ checkpoint mới nhất trong paths.checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Ghi đè seed.")
    parser.add_argument(
        "--total-steps", type=int, default=None, help="Ghi đè training.total_steps.",
    )
    return parser.parse_args()


def main() -> None:
    """Điểm vào của script."""
    args = parse_args()
    overrides: Dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.total_steps is not None:
        overrides["training.total_steps"] = args.total_steps

    config = load_config(args.config, overrides=overrides or None)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()


