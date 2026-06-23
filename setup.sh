#!/usr/bin/env bash
# 환경 세팅 스크립트
#
# 사용법:
#   Bare metal (예: A6000 호스트):
#     uv venv .venv --python 3.12 --prompt pid --seed
#     source .venv/bin/activate
#     bash setup.sh
#
#   Docker (NGC PyTorch 이미지 안, 예: B200 클라우드):
#     uv venv .venv --python 3.12 --system-site-packages --prompt pid --seed
#     source .venv/bin/activate
#     MLLM_SKIP_TORCH_INSTALL=1 bash setup.sh
#
set -euo pipefail

# === 사전 체크 ===

if ! command -v uv >/dev/null 2>&1; then
  cat <<'EOF' >&2
❌ uv가 설치돼 있지 않습니다.
설치:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  cat <<'EOF' >&2
❌ 가상환경이 활성화되지 않았습니다. 클러스터에서 노드 종류에 맞춰 venv 디렉토리 이름을 다르게 쓰세요. README 참고.
EOF
  exit 1
fi

echo "Venv:   $VIRTUAL_ENV"
echo "Python: $(which python) ($(python --version 2>&1))"
echo ""

PYDEXPI_COMMIT="83e5a11"

TORCH_INDEX="${TORCH_INDEX:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === 0. PyTorch (노드별 CUDA wheel) ===
echo "=== [0/3] PyTorch ==="
if [ "${MLLM_SKIP_TORCH_INSTALL:-0}" = "1" ]; then
  echo "  Skipping install (MLLM_SKIP_TORCH_INSTALL=1)."
  echo "  Will use existing PyTorch from base image."
else
  if [ -z "$TORCH_INDEX" ]; then
    cat <<'EOF' >&2
❌ TORCH_INDEX가 설정되지 않았습니다.

노드별로 다음 중 하나 export 후 다시 실행:
  A6000 (driver 550):     export TORCH_INDEX=https://download.pytorch.org/whl/cu124
  Blackwell (driver 570): export TORCH_INDEX=https://download.pytorch.org/whl/cu128
  최신 driver (580+):     export TORCH_INDEX=https://download.pytorch.org/whl/cu130

또는 .env에 TORCH_INDEX= 줄 추가.
EOF
    exit 1
  fi
  echo "  Installing torch torchvision torchaudio from $TORCH_INDEX..."
  uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# 검증: torch가 있는지
python <<'PY'
import sys
try:
    import torch
except ImportError:
    print("❌ PyTorch import 실패")
    sys.exit(1)

print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA build: {torch.version.cuda}")
print(f"  cuDNN: {torch.backends.cudnn.version()}")
print(f"  CUDA available: {torch.cuda.is_available()}")
PY

# === 1. 우리 repo (editable) ===
echo ""
echo "=== [1/3] Installing this repo ==="
uv pip install -e .

# === 2. pyDEXPI ===
echo ""
echo "=== [2/3] pyDEXPI ==="
mkdir -p third_party
if [ ! -d "third_party/pyDEXPI" ]; then
  git clone https://github.com/process-intelligence-research/pyDEXPI.git third_party/pyDEXPI
fi
(
  cd third_party/pyDEXPI
  git fetch
  git checkout "$PYDEXPI_COMMIT"
)
uv pip install -e "third_party/pyDEXPI"

# === 3. flash-attn ===
# 검증 매트릭스:
#   cu124 (A6000) + torch 2.6.0 → 2.7.4.post1 ✅
#   cu128 (Blackwell) + torch 2.11.0 → 2.7.4.post1 ✅
# 안 맞으면 .env에 FLASH_ATTN_VERSION override.
# 기본값 2.7.4.post1, 환경변수로 override 가능.
echo ""
echo "=== [3/3] flash-attn ==="
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
if [ "${MLLM_SKIP_FLASH_ATTN:-0}" = "1" ]; then
  echo "  Skipping (MLLM_SKIP_FLASH_ATTN=1)."
elif python -c "import flash_attn" 2>/dev/null; then
  ver=$(python -c "import flash_attn; print(flash_attn.__version__)")
  echo "  Already installed: flash_attn $ver"
else
  echo "  Installing flash-attn==$FLASH_ATTN_VERSION..."

  # 사전 청소: 옛 잔재 / 캐시가 ABI mismatch의 흔한 원인
  echo "  Cleaning previous flash-attn artifacts..."
  uv pip uninstall flash-attn 2>/dev/null || true
  rm -rf "$VIRTUAL_ENV"/lib/python*/site-packages/flash_attn* 2>/dev/null || true
  uv cache clean 2>/dev/null || true

  # flash-attn build requires torch to be available in the current env.
  "$VIRTUAL_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VIRTUAL_ENV/bin/python" -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation --no-cache

  # ABI mismatch 즉시 잡기 — 학습 중에 터지는 거보다 여기서 멈추는 게 나음
  python -c "import flash_attn" || {
    echo "❌ flash-attn import 실패 (ABI mismatch 가능성). PyTorch 버전과 매칭되는 flash-attn wheel을 못 찾음."
    echo "   확인:"
    echo "     1. GPU 노드에서 실행 중인지 (nvidia-smi)"
    echo "     2. conda (base) 비활성화됐는지 (conda deactivate)"
    echo "     3. FLASH_ATTN_VERSION 다른 값 시도 (예: 2.7.3, 2.8.3)"
    exit 1
  }
  echo "  ✅ flash_attn $(python -c 'import flash_attn; print(flash_attn.__version__)') verified"
fi