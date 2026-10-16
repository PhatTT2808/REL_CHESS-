#!/usr/bin/env python
"""Pretrain Behavioral Cloning (warm-start) trước PPO self-play.

Vì sao cần (xem báo cáo sau vòng PPO 1M thuần):
    - Không gian hành động 8100 nước; PPO từ khởi tạo ngẫu nhiên mất hàng
      triệu bước chỉ để học "không thua" (hòa 77%), chưa học "chiếu hết".
    - Dataset public ``ysong18/xiangqi-dataset`` (pipeline Pikafish, 63k vị trí
      train) cho biết nước đi tốt của engine ở mỗi thế cờ. Học bắt chước
      (CrossEntropy + MSE value) 1-2 giờ giúp policy có điểm xuất phát tốt,
      sau đó PPO self-play chỉ việc tinh chỉnh.

Vẫn là RL: checkpoint pretrain chỉ là điểm khởi tạo cho ``scripts/train_rl.py
--resume``; toàn bộ bài học chiến thuật vẫn do PPO tự chơi mà có.

Ví dụ:
    python scripts/pretrain.py --config configs/pretrain.yaml
    python scripts/pretrain.py --config configs/pretrain.yaml --max-samples 2000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.policy_net import XiangqiPolicyNet
from src.training.bc_dataset import DATASET_URL, XiangqiBCDataSet, bc_collate
from src.utils.checkpoint import save_checkpoint
from src.utils.config import ensure_dir, get_device, load_config, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger("pretrain")

def build_model_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Trích tham số kiến trúc model từ config (giống train_rl.py)."""
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


def download_dataset(path: Path) -> None:
    """Tải dataset parquet từ HuggingFace nếu chưa có (~46MB)."""
    import urllib.request

    ensure_dir(str(path.parent))
    logger.info("Đang tải dataset: %s", DATASET_URL)
    with urllib.request.urlopen(DATASET_URL, timeout=300) as resp, open(
        path, "wb"
    ) as fh:
        fh.write(resp.read())
    logger.info("Đã tải xong: %s", path)


def accuracy(logits: torch.Tensor, acts: torch.Tensor, mask: torch.Tensor) -> float:
    """Tỉ lệ dự đoán đúng nước (mẫu mask rỗng vẫn tính argmax như model)."""
    pred = logits.argmax(dim=-1)
    return float((pred == acts).float().mean().item()) if pred.numel() else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain BC cho cờ tướng")
def pretrain(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Chạy vòng lặp Behavioral Cloning và lưu ``checkpoints/pretrain.pt``."""
    set_seed(int(config.get("seed", 42)))
    bc = config["pretrain"]

    dataset_path = resolve_path(bc["dataset_path"])
    if args.download or not dataset_path.is_file():
        download_dataset(dataset_path)

    full = XiangqiBCDataSet(
        str(dataset_path),
        label=str(bc.get("label", "played")),
        max_samples=args.max_samples,
    )
    if len(full) < 10:
        raise RuntimeError("Dataset quá nhỏ — kiểm tra lại file parquet.")

    n_val = max(1, int(len(full) * float(bc.get("val_ratio", 0.05))))
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Chia tập: train=%d val=%d", len(train_set), len(val_set))

    batch_size = int(bc.get("batch_size", 256))
    nw = int(bc.get("num_workers", 2))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=bc_collate, num_workers=nw,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=bc_collate, num_workers=nw,
    )

    device = get_device(str(bc.get("device", config["training"]["device"])))
    model = XiangqiPolicyNet(**build_model_kwargs(config)).to(device)
    logger.info("Tham số: %s | device=%s", f"{model.num_parameters:,}", device)

    opt = torch.optim.Adam(
        model.parameters(), lr=float(bc.get("lr", 1e-3)), weight_decay=1e-4
    )
    epochs = int(bc.get("epochs", 8))
    value_coef = float(bc.get("value_coef", 0.5))
    out_dir = ensure_dir(config["paths"]["checkpoints"])
    ckpt_path = out_dir / "pretrain.pt"

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_p = run_v = run_acc = 0.0
        n_seen = 0
        for batch in train_loader:
            graphs = batch["batch"].to(device)
            mask = batch["legal_mask"].to(device)
            actions = batch["action"].to(device)
            v_tgt = batch["value_target"].to(device)
            weight = batch["weight"].to(device)

            out = model(graphs, mask)
            logp = F.log_softmax(out.logits, dim=-1)
            p_elem = -logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            w_sum = float(weight.sum())
            p_loss = ((p_elem * weight).sum() / weight.sum()
                      if w_sum > 0 else p_elem.mean())
            v_loss = F.mse_loss(out.value, v_tgt)
            loss = p_loss + value_coef * v_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = actions.size(0)
            run_p += float(p_loss.item()) * bs
            run_v += float(v_loss.item()) * bs
            run_acc += accuracy(out.logits, actions, mask) * bs
            n_seen += bs

        model.eval()
        vp = vv = vacc = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                graphs = batch["batch"].to(device)
                mask = batch["legal_mask"].to(device)
                actions = batch["action"].to(device)
                v_tgt = batch["value_target"].to(device)
                out = model(graphs, mask)
                bs = actions.size(0)
                vp += float(F.cross_entropy(out.logits, actions).item()) * bs
                vv += float(F.mse_loss(out.value, v_tgt).item()) * bs
                vacc += accuracy(out.logits, actions, mask) * bs
                n_v += bs
        vp /= max(n_v, 1)
        vv /= max(n_v, 1)
        vacc /= max(n_v, 1)

        logger.info(
            "epoch %2d/%d | train p=%.4f v=%.4f acc=%.3f | "
            "val p=%.4f v=%.4f acc=%.3f | %.1fs",
            epoch, epochs, run_p / max(n_seen, 1), run_v / max(n_seen, 1),
            run_acc / max(n_seen, 1), vp, vv, vacc, time.time() - t0,
        )

        if vacc > best_val_acc:
            best_val_acc = vacc
            save_checkpoint(
                model, opt, epoch, config["paths"]["checkpoints"],
                filename="pretrain.pt",
                extra={
                    "stage": "pretrain_bc", "epoch": epoch, "val_acc": vacc,
                    "dataset": str(dataset_path),
                },
            )
            logger.info("  -> val acc tốt nhất (%.3f), lưu %s", vacc, ckpt_path)

    logger.info("Hoàn tất pretrain. Val accuracy tốt nhất = %.3f", best_val_acc)
    logger.info(
        "Bước sau: dùng pretrain.pt làm điểm khởi tạo cho PPO, ví dụ đổi "
        "paths.checkpoints sang thư mục riêng rồi copy pretrain.pt vào đó và "
        "chạy: python scripts/train_rl.py --config configs/default.yaml --resume"
    )


def main() -> None:
    """Điểm vào của script."""
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    pretrain(config, args)


if __name__ == "__main__":
    main()
    p.add_argument("--config", default="configs/pretrain.yaml")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Giới hạn số mẫu (debug nhanh).")
    p.add_argument("--download", action="store_true",
                   help="Buộc tải lại dataset dù file đã tồn tại.")
    return p.parse_args()

