#!/usr/bin/env python
"""Script đánh giá agent: Random / Engine UCI / Pretrained skeleton.

Ba bài test (đúng thiết kế đồ án):
    1. vs Random (mặc định 100 ván).
    2. vs Engine độ sâu 1–2 (bỏ qua an toàn nếu thiếu engine binary).
    3. vs "Pretrained skeleton" — checkpoint cũ hơn (mặc định 50 ván).

Kết quả in ra bảng markdown gồm Elo + khoảng tin cậy 95%.

Ví dụ:
    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint checkpoints_smoke/step_2252.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.elo import MatchSummary
from src.evaluation.match import EngineAgent, ModelAgent, RandomAgent, play_match
from src.models.policy_net import XiangqiPolicyNet
from src.utils.checkpoint import list_checkpoints, load_checkpoint
from src.utils.config import get_device, load_config, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger("evaluate")


def setup_logging(log_dir: str) -> None:
    """Cấu hình logging ra console + file."""
    from src.utils.config import ensure_dir

    directory = ensure_dir(log_dir)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(directory / "evaluate.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


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


def load_model(config: Dict[str, Any], checkpoint: str, device) -> XiangqiPolicyNet:
    """Nạp model từ checkpoint."""
    model = XiangqiPolicyNet(**build_model_kwargs(config))
    load_checkpoint(checkpoint, model, map_location=str(device))
    return model.to(device).eval()


def find_pretrained_skeleton(
    config: Dict[str, Any], exclude: List[Path]
) -> Optional[Path]:
    """Tìm một checkpoint "pretrained skeleton" cũ hơn để làm đối thủ.

    Ưu tiên checkpoint trong pool có số bước nhỏ nhất (chưa huấn luyện nhiều).
    """
    candidates: List[tuple[int, Path]] = []
    for directory in (config["paths"]["checkpoints"], config["paths"]["pool"]):
        for step, path in list_checkpoints(directory):
            if path.resolve() not in {p.resolve() for p in exclude}:
                candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def evaluate_all(
    config: Dict[str, Any],
    checkpoint: str,
    device,
    num_games: Optional[int] = None,
    include_engine: bool = True,
) -> List[tuple[str, MatchSummary]]:
    """Chạy 3 bài test và trả về danh sách ``(nhãn, MatchSummary)``."""
    ev = config["evaluation"]
    train_cfg = config["training"]
    env_cfg = config["env"]
    n_games = num_games or int(ev["num_games"])
    max_moves = int(env_cfg["max_moves"])
    perp = int(env_cfg["perpetual_check_limit"])
    lmf = bool(env_cfg["legal_move_filter"])
    seed = int(config.get("seed", 42))

    results: List[tuple[str, MatchSummary]] = []
    model = load_model(config, checkpoint, device)
    agent = ModelAgent(model, device=device, deterministic=True, name="Model")

    # --- Bài test 1: vs Random ---
    logger.info("Test 1/%d: đấu với Random (%d ván)", 3, n_games)
    random_agent = RandomAgent(seed=seed, name="Random")
    summary = play_match(
        agent, random_agent, num_games=n_games, max_moves=max_moves,
        perpetual_check_limit=perp, legal_move_filter=lmf,
        opponent_elo=0.0, elo_k=float(ev["elo_k"]),
        bootstrap_samples=int(ev["bootstrap_samples"]), seed=seed,
        label="vs Random",
    )
    results.append(("vs Random", summary))

    # --- Bài test 2: vs Engine UCI/UCCI (bỏ qua an toàn nếu thiếu binary) ---
    if include_engine:
        engine_path = ev.get("engine_path", "") or ""
        protocol = ev.get("engine_protocol", "uci")
        depth = int(ev["engine_depth"])
        threads = int(ev["engine_threads"])
        if engine_path and resolve_path(engine_path).is_file():
            try:
                logger.info(
                    "Test 2/3: đấu với engine %s (depth=%d, %d ván)",
                    protocol, depth, n_games,
                )
                engine_agent = EngineAgent(
                    str(resolve_path(engine_path)), protocol=protocol,
                    depth=depth, threads=threads, name=f"Engine-{protocol}",
                )
                summary = play_match(
                    agent, engine_agent, num_games=n_games, max_moves=max_moves,
                    perpetual_check_limit=perp, legal_move_filter=lmf,
                    opponent_elo=1800.0, elo_k=float(ev["elo_k"]),
                    bootstrap_samples=int(ev["bootstrap_samples"]), seed=seed,
                    label=f"vs Engine depth {depth}",
                )
                results.append((f"vs Engine d{depth}", summary))
                engine_agent.close()
            except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
                logger.warning("Bỏ qua bài test engine do lỗi: %s", exc)
        else:
            logger.warning(
                "Bỏ qua bài test engine: chưa cấu hình/không tìm thấy "
                "'evaluation.engine_path'. Đặt đường dẫn tới Pikafish/Eleeye để bật."
            )

    # --- Bài test 3: vs Pretrained skeleton (checkpoint cũ hơn) ---
    skeleton_path = find_pretrained_skeleton(config, exclude=[resolve_path(checkpoint)])
    if skeleton_path is not None:
        n_skeleton = max(1, int(ev["num_games"]) // 2) or 50
        logger.info(
            "Test 3/3: đấu với pretrained skeleton %s (%d ván)",
            skeleton_path.name, n_skeleton,
        )
        skeleton_model = load_model(config, str(skeleton_path), device)
        skeleton_agent = ModelAgent(
            skeleton_model, device=device, deterministic=True,
            name=f"Skeleton({skeleton_path.stem})",
        )
        summary = play_match(
            agent, skeleton_agent, num_games=n_skeleton, max_moves=max_moves,
            perpetual_check_limit=perp, legal_move_filter=lmf,
            opponent_elo=0.0, elo_k=float(ev["elo_k"]),
            bootstrap_samples=int(ev["bootstrap_samples"]), seed=seed,
            label="vs Pretrained skeleton",
        )
        results.append(("vs Pretrained skeleton", summary))
    else:
        logger.warning(
            "Bỏ qua bài test skeleton: không tìm thấy checkpoint đối thủ khác."
        )

    return results


def render_table(results: List[tuple[str, MatchSummary]]) -> str:
    """Tạo bảng markdown từ danh sách kết quả."""
    lines = [MatchSummary.table_header()]
    lines.append("|" + "-" * 30 + "|" + "-" * 7 + "|" + "-" * 7 + "|"
                 + "-" * 7 + "|" + "-" * 8 + "|" + "-" * 10 + "|"
                 + "-" * 11 + "|" + "-" * 27 + "|")
    for label, summary in results:
        lines.append(summary.as_row(label))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Đánh giá agent cờ tướng")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Đường dẫn file cấu hình YAML.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint cần đánh giá (mặc định: mới nhất trong paths.checkpoints).",
    )
    parser.add_argument(
        "--num-games", type=int, default=None, help="Ghi đè số ván mỗi bài test.",
    )
    parser.add_argument(
        "--no-engine", action="store_true", help="Bỏ qua bài test engine.",
    )
    return parser.parse_args()


def main() -> None:
    """Điểm vào của script đánh giá."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    setup_logging(config["paths"]["logs"])

    device = get_device(config["training"]["device"])

    checkpoint = args.checkpoint
    if checkpoint is None:
        items = list_checkpoints(config["paths"]["checkpoints"])
        if not items:
            logger.error(
                "Không tìm thấy checkpoint trong '%s'. Hãy huấn luyện trước.",
                config["paths"]["checkpoints"],
            )
            sys.exit(1)
        checkpoint = str(items[-1][1])

    logger.info("=" * 70)
    logger.info("Đánh giá agent: %s | device=%s", checkpoint, device)
    logger.info("=" * 70)

    results = evaluate_all(
        config, checkpoint, device,
        num_games=args.num_games, include_engine=not args.no_engine,
    )

    table = render_table(results)
    print("\n" + table + "\n")

    # Lưu bảng kết quả ra file.
    out_path = resolve_path(config["paths"]["logs"]) / "eval_results.md"
    out_path.write_text(table, encoding="utf-8")
    logger.info("Đã lưu bảng kết quả: %s", out_path)


if __name__ == "__main__":
    main()


