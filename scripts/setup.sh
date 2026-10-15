#!/usr/bin/env bash
# =====================================================================
# setup.sh — Cài đặt dự án Xiangqi RL trên máy Linux (VPS/GPU thuê)
# ---------------------------------------------------------------------
# Tạo venv (không cần conda) và cài toàn bộ thư viện.
#
# Dùng:
#   bash scripts/setup.sh              # cài bản CPU
#   bash scripts/setup.sh cu124        # cài bản CUDA 12.4
#   bash scripts/setup.sh cu121        # cài bản CUDA 12.1
#
# Sau khi xong:
#   source .venv/bin/activate
# =====================================================================

set -euo pipefail

# --- 0. Xác định thư mục gốc dự án (chạy được từ bất kỳ đâu) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo ">> Thư mục dự án: ${PROJECT_ROOT}"

# --- 1. Chọn biến thể torch (cpu / cu121 / cu124 ...) ---
TORCH_VARIANT="${1:-cpu}"
if [ "${TORCH_VARIANT}" = "cpu" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_VARIANT}"
fi
echo ">> Biến thể torch: ${TORCH_VARIANT} (${TORCH_INDEX})"

# --- 2. Tìm Python >= 3.10 (ưu tiên 3.12) ---
PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "${cand}" >/dev/null 2>&1; then
    PYTHON_BIN="${cand}"
    break
  fi
done
if [ -z "${PYTHON_BIN}" ]; then
  echo "!! Không tìm thấy Python. Cài bằng:  sudo apt install -y python3.12 python3.12-venv" >&2
  exit 1
fi
echo ">> Dùng Python: $(command -v "${PYTHON_BIN}") ($(${PYTHON_BIN} --version))"

# --- 3. Tạo venv (có fallback khi thiếu ensurepip) ---
if [ ! -d ".venv" ]; then
  echo ">> Tạo virtual environment .venv ..."
  if ! "${PYTHON_BIN}" -m venv .venv 2>/dev/null; then
    echo "!! 'python -m venv' thất bại (thiếu ensurepip). Thử lại không cần pip ..."
    "${PYTHON_BIN}" -m venv --without-pip .venv
    # Bootstrap pip bằng get-pip.py
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py --quiet
  fi
else
  echo ">> .venv đã tồn tại, bỏ qua bước tạo."
fi

VENV_PY=".venv/bin/python"

# --- 4. Nâng cấp pip ---
echo ">> Nâng cấp pip ..."
"${VENV_PY}" -m pip install --quiet --upgrade pip

# --- 5. Cài torch trước (đúng biến thể CPU/CUDA) ---
echo ">> Cài torch (${TORCH_VARIANT}) ..."
"${VENV_PY}" -m pip install --quiet torch --index-url "${TORCH_INDEX}"

# --- 6. Cài phần còn lại từ requirements.txt ---
# Dùng --upgrade + --extra-index-url: requirements.txt khai báo ``torch>=2.2``.
# Nếu KHÔNG có --upgrade, pip thấy torch đã cài (ở bước 5) nên sẽ GIỮ bản CPU
# và ghi đè bản CUDA vừa cài => ``torch.cuda.is_available() == False``.
echo ">> Cài requirements.txt ..."
"${VENV_PY}" -m pip install --quiet --upgrade -r requirements.txt \
  --extra-index-url "${TORCH_INDEX}"

# Bảo đảm torch đúng biến thể: nếu requirements.txt kéo lại bản CPU thì ép lại.
if ! "${VENV_PY}" - <<'PY'
import sys, torch
sys.exit(0 if ("+cpu" not in torch.__version__) else 1)
PY
then
  echo ">> requirements.txt đã kéo torch về bản CPU -> cài lại bản ${TORCH_VARIANT}"
  "${VENV_PY}" -m pip install --quiet --force-reinstall torch --index-url "${TORCH_INDEX}"
fi

# --- 7. Kiểm tra ---
echo ">> Kiểm tra import ..."
"${VENV_PY}" - <<'PY'
import importlib.metadata as md
import torch, torch_geometric, cchess, numpy, matplotlib, yaml, pytest
print("  torch           :", torch.__version__)
print("  torch_geometric :", torch_geometric.__version__)
print("  cchess          :", md.version("cchess"))
print("  numpy           :", numpy.__version__)
print("  matplotlib      :", matplotlib.__version__)
print("  PyYAML          :", md.version("PyYAML"))
print("  pytest          :", pytest.__version__)
print("  CUDA available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU             :", torch.cuda.get_device_name(0))
PY

echo ""
echo "======================================================================"
echo " CÀI ĐẶT XONG. Kích hoạt môi trường bằng:"
echo "     source .venv/bin/activate"
echo " Rồi chạy thử nhanh:"
echo "     python -m pytest -q"
echo "     python scripts/train_rl.py --config configs/smoke.yaml"
echo "======================================================================"
