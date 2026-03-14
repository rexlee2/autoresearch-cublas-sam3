#!/usr/bin/env bash
# =========================================================================
# Quick setup for running SAM3 GEMM benchmark on AWS EC2 GPU instance.
#
# Usage:
#   1. Launch a g5.xlarge (A10G, 24GB, ~$1/hr) or g4dn.xlarge (T4, ~$0.53/hr)
#      with a Deep Learning AMI (PyTorch)
#   2. SSH in and run: bash setup_aws.sh
#   3. Then: ./run.sh
#
# Recommended AMI: "Deep Learning AMI GPU PyTorch" (comes with PyTorch + CUDA)
# =========================================================================

set -euo pipefail

echo "=== SAM3 GEMM Autoresearch Setup ==="

# 1. Check GPU
echo ""
echo "--- Checking GPU ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  Backend: NVIDIA / cuBLASLt"
elif command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname
    echo "  Backend: AMD / hipBLASLt"
else
    echo "ERROR: No GPU detected (no nvidia-smi or rocm-smi)"
    exit 1
fi

# 2. Check PyTorch
echo ""
echo "--- Checking PyTorch ---"
if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "  PyTorch OK"
else
    echo "  PyTorch not found. Installing..."
    pip install torch
fi

# 3. Verify CUDA/ROCm matmul works
echo ""
echo "--- Quick GEMM sanity check ---"
python3 -c "
import torch
a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
b = torch.randn(256, 256, device='cuda', dtype=torch.float16)
c = torch.matmul(a, b)
print(f'  GEMM OK: {c.shape}, device={c.device}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# 4. Init git repo if needed
echo ""
echo "--- Git setup ---"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    cd ..
    git init
    git add -A
    git commit -m "initial SAM3 GEMM autoresearch setup"
fi
echo "  Git OK"

# 5. Memory check for largest shape
echo ""
echo "--- VRAM check for SAM3 shapes ---"
python3 -c "
import torch
# Largest shape: (5184, 36288, 256)
# A: 5184*256*2 = 2.65 MB, B: 256*36288*2 = 18.6 MB, C: 5184*36288*2 = 377 MB
# Total ~400 MB + workspace.  Should fit in 16+ GB.
free_mem = torch.cuda.mem_get_info()[0] / 1e9
total_mem = torch.cuda.mem_get_info()[1] / 1e9
needed_mb = 400 + 128  # largest shape + workspace
print(f'  GPU memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total')
print(f'  Largest SAM3 shape needs ~{needed_mb} MB — {\"OK\" if free_mem > needed_mb/1000 else \"WARNING: tight\"}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run:"
echo "  cd $(pwd)"
echo "  ./run.sh                 # baseline benchmark"
echo "  ./run.sh --loop 10       # agent iterates 10 times"
echo ""
echo "Estimated cost: <\$0.10 for a full run on g5.xlarge (~\$1/hr)"
