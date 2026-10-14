# Xiangqi RL — Neural-Symbolic Reinforcement Learning cho Cờ Tướng

Đồ án môn **REL301m**: xây dựng agent tự học chơi Cờ Tướng (Xiangqi) từ đầu bằng
**PPO Self-Play**, dùng **GNN (Graph Attention Network)** làm state encoder kết
hợp **Tactical Motif Attention** — kiến trúc **Neural-Symbolic** (mạng nơ-ron +
tri thức ký hiệu về cấu trúc tấn công/phòng thủ và các "mô-típ" chiến thuật).

Không dùng dataset ngoài, không gọi API trả phí trong training loop.

---

## 1. Kiến trúc

```
Thế cờ (FEN)
   │
   ├─► Graph Builder ──► PyG Data (mỗi quân = 1 node, cạnh tấn công/phòng thủ/cùng đường)
   │
   ├─► GNN Encoder (GATConv ×3, hidden=128, heads=4, BatchNorm+ReLU+residual)
   │        └─► graph_emb (128)          [global_mean_pool]
   │
   ├─► Motif Attention (Cross-attention: Query=graph_emb, Key/Value=Motif Bank 30×64)
   │        └─► tactical_context (64) + attn_weights (30)   ← trực quan hóa được
   │
   └─► Concat [graph_emb(128) | tactical_context(64)] = 192
            ├─► Policy Head (192→512→512→8100)  + legal mask
            └─► Value Head  (192→256→1) + tanh
```

- **Action space 8100** = 90 ô × 90 ô; `action = from_idx*90 + to_idx`, `idx = y*9 + x`.
- **Legal mask** che các nước không hợp lệ (bao gồm cả nước tự-chiếu-tướng).
- **Motif Bank** là `nn.Parameter(30, 64)` học được: attention weights cho biết
  "mô-típ chiến thuật nào đang kích hoạt" ở mỗi nước — phần *symbolic* có thể giải thích.

---

## 2. Cài đặt — máy Linux mới (venv, không cần conda)

> Hướng dẫn này dùng **`venv`** (có sẵn trong Python, nhẹ, không cần cài gì thêm).
> Mọi lệnh dưới đây **dán nguyên vào terminal** mới trên máy Linux (Ubuntu/VPS/GPU
> thuê) theo đúng thứ tự. Nếu máy bạn đã cài conda thì vẫn dùng được (xem mục 2.4).

### 2.0. Yêu cầu đầu vào
| Hạng mục | Cần có | Cách kiểm tra |
|---|---|---|
| Kết nối mạng | có (để `pip` tải gói) | `ping -c 1 pypi.org` |
| Quyền sudo | có (để `apt install` nếu thiếu) | `sudo -n true` |
| Terminal | SSH hoặc shell Linux | `uname -a` → có chữ `Linux` |

### 2.1. Chuẩn bị hệ thống (mới hoàn toàn — chạy 1 lần)

```bash
# 1. Cập nhật danh sách gói
sudo apt update

# 2. Cài Python 3.12 + venv + build tool (thiếu cái này sẽ báo lỗi ensurepip)
sudo apt install -y python3 python3.12 python3.12-venv python3-pip build-essential curl git

# 3. Xác nhận
python3.12 --version          # => Python 3.12.x
python3.12 -m venv --help >/dev/null && echo "venv OK"
```

> ⚠️ **Lỗi thường gặp:** nếu bỏ bước `python3.12-venv` sẽ gặp
> `ensurepip is not available` → đây là lỗi **bắt buộc** phải sửa trước khi tiếp.

### 2.2. Tạo venv + cài thư viện (2 cách — chọn 1)

**Cách A — nhanh (khuyên dùng), chạy script tự động:**

```bash
# 1. Lấy mã nguồn dự án (nếu chưa có)
git clone <repo-url> Project_REL && cd Project_REL
# (hoặc copy thư mục Project_REL vào máy rồi cd Project_REL)

# 2. Cài bản CPU — một lệnh là xong
bash scripts/setup.sh

#   … hoặc cài bản CUDA 12.4 (nếu thuê GPU):
bash scripts/setup.sh cu124

# 3. Kích hoạt môi trường
source .venv/bin/activate
```

**Cách B — tay từng bước (hiểu rõ hơn, debug dễ):**

```bash
cd Project_REL

# 1. Tạo venv
python3.12 -m venv .venv

# 2. Bật venv (mọi lệnh sau đều nằm trong môi trường cô lập)
source .venv/bin/activate          # sau bước này terminal hiện (…)

# 3. Nâng cấp pip
python -m pip install --upgrade pip

# 4. Cài PyTorch — CHỌN 1 TRONG 2 (chọn đúng CPU/GPU của máy)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124    # CUDA 12.4

# 5. Cài nốt phần còn lại từ requirements.txt
python -m pip install -r requirements.txt
```

### 2.3. Kiểm tra cài đặt (bắt buộc chạy trước khi train)

```bash
source .venv/bin/activate

# 1. Import toàn bộ thư viện
python -c "import torch, torch_geometric, cchess, numpy, matplotlib, yaml, pytest; \
print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cchess OK')"

# 2. Nếu cài CUDA: xác nhận GPU đã nhận
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 3. Chạy toàn bộ unit test (kỳ vọng: 38 passed)
python -m pytest -q
```

**Kết quả mong đợi:**
```
torch 2.x.x | pyg 2.8.x | cchess OK
CUDA: True NVIDIA Tesla T4        # (hoặc False nếu bản CPU)
......................................                                     [100%]
38 passed
```

### 2.4. (Tuỳ chọn) Nếu bạn vẫn muốn dùng conda

```bash
conda create -y -n REL_chess python=3.12
conda activate REL_chess
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # hoặc /cpu
python -m pip install -r requirements.txt
```

> **Lưu ý:** từ `torch_geometric >= 2.3` không cần `torch-scatter`/`torch-sparse`;
> `GATConv` chạy thuần PyTorch. Mọi script đều chạy được cả CPU lẫn GPU
> (`training.device: auto`).

---

## 3. Cấu trúc thư mục

```
Project_REL/
├── src/
│   ├── env/           constants.py, graph_builder.py, xiangqi_env.py
│   ├── models/        gnn_encoder.py, motif_attention.py, policy_net.py
│   ├── training/      reward.py, ppo.py, self_play.py
│   ├── evaluation/    elo.py, match.py
│   └── utils/         config.py, seed.py, checkpoint.py
├── configs/           default.yaml (GPU production), smoke.yaml (test nhanh CPU)
├── scripts/           setup.sh, train_rl.py, evaluate.py, visualize.py
├── tests/             conftest.py, test_env.py, test_model.py, test_ppo.py
├── requirements.txt
└── README.md
```

---

## 4. Sử dụng

### 4.1. Huấn luyện

```bash
# Chạy thử nhanh trên CPU (kiểm chứng pipeline, ~10-30 giây)
python scripts/train_rl.py --config configs/smoke.yaml

# Chạy đầy đủ (1e6 steps, 512 games/batch) — dùng trên GPU Linux
python scripts/train_rl.py --config configs/default.yaml

# Tiếp tục từ checkpoint mới nhất
python scripts/train_rl.py --config configs/default.yaml --resume

# Ghi đè nhanh một số tham số
python scripts/train_rl.py --config configs/default.yaml --total-steps 200000 --seed 7
```

### 4.2. Đánh giá

```bash
# 3 bài test: vs Random, vs Engine UCI (nếu có), vs Pretrained skeleton
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/step_1000000.pt

# Bỏ qua bài test engine
python scripts/evaluate.py --config configs/default.yaml --no-engine
```

Kết quả in ra bảng markdown (Elo + CI 95%) và lưu vào `logs/eval_results.md`.

### 4.3. Trực quan hóa

```bash
python scripts/visualize.py --config configs/default.yaml
```
Sinh ra trong `figures/`:
- `metrics_curves.png` — policy/value loss, entropy, win-rate.
- `elo_curve.png` — Elo theo checkpoint (nếu có `logs/elo_history.jsonl`).
- `attention_heatmap.png` — motif nào kích hoạt ở mỗi nước đi.
- `motif_usage.png` — mức sử dụng trung bình từng motif.

### 4.4. Kiểm thử

```bash
python -m pytest -q          # 38 tests
```

---

## 5. Chọn GPU/CPU để thuê (quan trọng — đọc trước khi thuê)

> **Kết luận nhanh:** dự án này **rất nhẹ về GPU** (model chỉ ~4.6M tham số,
> dùng <1 GB VRAM). Nút cổ chai thật là **CPU** (vì `cchess` là pure Python) và
> **RAM**. Vì vậy **đừng thuê GPU đắt tiền** — hãy ưu tiên **nhiều CPU core + RAM lớn**.

### Vì sao?
| Thành phần | Quy mô |
|---|---|
| Node/ván | ≤ 32 quân |
| GNN (GAT ×3, hidden 128) | ~0.7M params |
| Policy head (192→512→512→8100) | ~4.6M params |
| **Tổng** | **~4.6M params** |

- Mỗi ván 200 nước, mỗi nước phải lọc legal mask bằng `is_checked_move()`
  (copy board + `is_checking`) → tốn CPU đáng kể.
- GNN forward/backward cho batch 512 ván chỉ tốn vài chục ms trên GPU bất kỳ.
- Nếu CPU không kịp sinh self-play, GPU sẽ **idle** và bạn trả tiền vô ích.

### Cấu hình khuyến nghị (tiết kiệm nhất)
| Cấu hình | Nhà cung cấp | Giá tham khảo | Đánh giá |
|---|---|---|---|
| **8–16 vCPU + 32–64 GB RAM + T4/RTX 3060/4060** | Vast.ai, RunPod | ~$0.15–0.35/h | ⭐ **Tối ưu nhất cho dự án này** |
| 4–8 vCPU + 16–32 GB RAM + T4 16GB | Vast.ai community | ~$0.10–0.20/h | Rẻ nhất, vẫn dư sức |
| 8–16 vCPU + 32–64 GB RAM (**không GPU**) | RunPod CPU / Hetzner | ~$0.10–0.25/h | Tiết kiệm tối đa, chậm hơn chút |
| RTX 3060 12GB (mua máy) | — | ~5–7tr (cũ) | ⭐ Sweet spot nếu mua |
| ~~A100 / H100 / A6000~~ | — | rất đắt | ❌ **Lãng phí hoàn toàn** |

**Quy đổi nhanh:** với `default.yaml` (1e6 steps), trên máy 8–16 vCPU dự kiến
**~1–3 giờ** → tổng chi phí chỉ **vài xu đến ~$1**.

### Quy trình thuê GPU Linux & chạy (từng bước, từ terminal mới)

> Sau khi đăng nhập SSH vào máy thuê, **dán theo đúng thứ tự** các lệnh sau.

**Bước 1 — Chuẩn bị hệ thống**
```bash
sudo apt update
sudo apt install -y python3 python3.12 python3.12-venv python3-pip build-essential curl git tmux
```

**Bước 2 — Lấy mã nguồn & cài môi trường**
```bash
# Lấy mã (clone hoặc copy vào máy, rồi vào thư mục)
git clone <repo-url> Project_REL && cd Project_REL

# Cài một lệnh (tự tạo venv + cài torch đúng biến thể + requirements)
bash scripts/setup.sh cu124        # GPU CUDA 12.4
# bash scripts/setup.sh            # nếu là máy CPU-only

# Kích hoạt môi trường (mọi lệnh sau đều cần bước này)
source .venv/bin/activate
```

**Bước 3 — Kiểm tra môi trường & GPU**
```bash
# Import OK?
python -c "import torch, torch_geometric, cchess; print(torch.__version__, torch_geometric.__version__)"

# GPU đã nhận chưa?  (kỳ vọng: True + tên GPU)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Chạy test (kỳ vọng: 38 passed)
python -m pytest -q
```

**Bước 4 — Train bền (không mất khi ngắt SSH)**
```bash
mkdir -p logs

# Cách 1: tmux (khuyên dùng — xem lại được, detach/attach)
tmux new -s train
python scripts/train_rl.py --config configs/default.yaml 2>&1 | tee logs/train_console.log
# (Ctrl+B rồi D để detach;  tmux attach -t train  để xem lại;  Ctrl+B %  để chia màn hình)

# Cách 2: nohup (chạy nền, đóng SSH cũng không chết)
nohup python scripts/train_rl.py --config configs/default.yaml > logs/train_console.log 2>&1 &
tail -f logs/train_console.log        # Ctrl+C để thoát xem (train vẫn chạy)
```

**Bước 5 — Giám sát & hoàn tất**
```bash
# Script tự log ETA (giờ) mỗi batch → ước lượng chi phí thuê.
# GPU-Util thấp (<20%) → đang nghẽn CPU → tăng training.num_workers trong config.

nvidia-smi -l 5                       # giám sát GPU (nếu có)
ls -la checkpoints/ logs/             # xác nhận checkpoint + metrics.jsonl

# Tải kết quả về máy local (chạy từ MÁY CỦA BẠN, không phải máy thuê)
scp -P <port> user@host:~/Project_REL/checkpoints/*.pt ./checkpoints/
scp -P <port> user@host:~/Project_REL/figures/*.png ./figures/
```

- **Resume an toàn:** nếu bị ngắt, chạy lại với `--resume` để tiếp tục từ checkpoint gần nhất.
- **Giám sát GPU:** `nvidia-smi -l 5`. Nếu `GPU-Util` thấp (<20%) → đang nghẽn CPU
  → tăng `training.num_workers`.

---

## 6. Siêu tham số chính (`configs/default.yaml`)

| Nhóm | Giá trị |
|---|---|
| Model | GNN hidden 128, 3 lớp GAT (4 heads), motif 30×64, policy 512, value 256 |
| PPO | lr 2.5e-4 (cosine), γ=0.99, GAE λ=0.95, clip 0.2, vf 0.5, ent 0.01, grad-clip 0.5, 4 epochs |
| Batch | 512 games, max 200 moves, minibatch 4096 bước |
| Tổng | 1,000,000 steps, checkpoint mỗi 10,000 steps |
| Reward | thắng +1, thua −1, **hoà −0.1**, **step_penalty 0.005**, potential-based shaping |

### 6.1. Lưu ý: reward đã sửa sau vòng train đầu tiên

Vòng train 1e6 bước đầu (kết quả: 91/100 ván hoà, thua cả Random) cho thấy
**reward cũ sai**. Cụ thể:

- Shaping kiểu cũ `r_t = coef·Φ(s_t)` cộng dồn **~12.5 điểm/ván** (terminal chỉ ±1),
  khiến agent học "ăn cho nhiều rồi giằng co" thay vì chiếu hết.
- Value head có `tanh` kẹp trong [−1,1] nên không biểu diễn được return ±12.

Đã sửa:

1. **Potential-based shaping** (Ng et al., 1999): `r = γ·Φ(s') − Φ(s)`, bảo toàn
   chính sách tối ưu — tổng shaping không còn tích luỹ theo độ dài ván.
2. **Value head bỏ `tanh`** → thang giá trị tự do.
3. **Hoà = −0.1, step_penalty = 0.005** → phạt kéo giờ.
4. **Checkpoint giờ lưu scheduler + optimizer + RNG** → resume đúng tiến độ.

Đo sau khi sửa (smoke test): tổng reward/ván ≈ **−0.25…−0.42** (trước: ~±12),
`value_loss` giảm từ **1.3–1.7 → ~0.005–0.015**.

> ⚠️ **Checkpoint cũ không tương thích** (do bỏ `tanh` làm đổi kiến trúc value_head).
> Phải train lại từ đầu, không `--resume` từ checkpoint cũ.

---

## 7. Khác biệt quan trọng giữa `cchess` và API giả định ban đầu

Thư viện thực tế là **`cchess==1.25.5`** (không phải `python-chinese-chess`).
Mã nguồn đã được điều chỉnh để khớp API thật:

| Giả định ban đầu | Thực tế trong `cchess` 1.25.5 | Cách xử lý |
|---|---|---|
| `board.turn` | không có | dùng `board.get_move_color()` (RED=1/BLACK=2) |
| `board.is_game_over()` | **không có** | tự cài trong `xiangqi_env.py` |
| `board.is_repetition(3)` | **không có** | đếm khoá FEN (3-fold) |
| `board.piece_map()` | **không có** | `graph_builder.iter_pieces()` |
| `p.piece_type` | **không có** | dùng `Piece.species` / ký tự FEN |
| `board.move()` tự đổi lượt | **KHÔNG** tự đổi lượt | gọi `board.next_turn()` sau mỗi nước |
| `create_moves()` lọc tự-chiếu-tướng | **KHÔNG** lọc | lọc thêm bằng `is_checked_move()` |
| `board.zhash()` dùng để đếm lặp | **có bug** (hash không đổi sau vài nước) | dùng khoá FEN thay thế |
| `cchess.engine` cho UCI | có (`EngineManager.load_uci/load_ucci`) | bọc trong `EngineAgent`, tự bỏ qua nếu thiếu binary |

> ⚠️ Hai điểm dễ sai nhất đã được xử lý và có unit test bảo vệ:
> **(1)** `move()` không tự đổi lượt → nếu quên `next_turn()`, cả hai bên đều đi Đỏ;
> **(2)** legal mask phải lọc `is_checked_move()`, nếu không agent có thể đi nước
> để lộ tướng và `no_moves()` sẽ không phát hiện được hết cờ.

---

## 8. Unit tests (`python -m pytest -q`)

- **`test_env.py`** — reset/step/mask, mã hóa action, mask loại nước tự-chiếu,
  chiếu bí → kết thúc, mất tướng không crash, chạm trần nước → hoà, hình dạng đồ thị,
  tính tất định theo seed.
- **`test_model.py`** — forward shape `(B,8100)/(B,1)/(B,30)`, xác suất ngoài mask = 0,
  logits ngoài mask = `NEG_INF`, `act` chỉ trả nước hợp lệ, `evaluate` entropy ≥ 0,
  **gradient chảy tới mọi tham số (kể cả `motif_bank`)**, kích thước head đúng spec,
  tính tất định.
- **`test_ppo.py`** — GAE khớp công thức tính tay, clipped surrogate bị clip đúng,
  `PPOConfig.from_dict`, `Trajectory`, `update` thay đổi tham số, scheduler giảm lr.

---

## 9. Tài liệu tham khảo
- PPO: Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- GAT: Veličković et al., *Graph Attention Networks* (ICLR 2018).
- GAE: Schulman et al., *High-Dimensional Continuous Control Using GAE* (ICLR 2016).
- PFSP: AlphaStar (Vinyals et al., 2019).
- Thư viện cờ: [`cchess`](https://github.com/walker8088/cchess) · [`torch_geometric`](https://pytorch-geometric.readthedocs.io)

