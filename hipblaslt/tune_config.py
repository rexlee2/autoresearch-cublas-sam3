"""
EDITABLE tuning configuration for GEMM optimization.

This is the ONLY file the AI agent should modify.

Target: Meta SAM3 (Segment Anything Model 3) inference.
Works on both NVIDIA (cuBLASLt) and AMD (hipBLASLt) GPUs.

Findings from this tuning are relevant to AMD/ROCm researchers because:
  1. SAM3 shapes are non-standard (M=5184, N=4736, N=36288, K=256)
  2. Both cuBLASLt and hipBLASLt have heuristics tuned for LLM shapes
  3. Dimension alignment, workspace size, and layout choices affect both
  4. The performance gaps found here will be LARGER on hipBLASLt
"""

import torch

# =========================================================================
# TUNING KNOBS — the agent modifies these to improve SAM3 GEMM throughput
# =========================================================================

# ---------------------------------------------------------------------------
# 1. MATMUL PRECISION
#
# Controls whether TF32 is used for float32 accumulation on Ampere+ GPUs.
# "highest" = pure FP32 accum (slowest, most precise)
# "high"    = TF32 for internal but FP32 result (balanced)
# "medium"  = TF32 throughout (fastest, least precise)
#
# For FP16 inputs this mainly affects the accumulator.  May change which
# cuBLASLt kernel is selected.
#
# AMD equivalent: hipBLASLt compute_type (f32_r vs f16_r)
# ---------------------------------------------------------------------------
MATMUL_PRECISION = "high"

# ---------------------------------------------------------------------------
# 2. ALLOW TF32
#
# Separate from precision — this is a global cuBLAS flag.
# AMD equivalent: HIPBLASLT_COMPUTE_TYPE
# ---------------------------------------------------------------------------
ALLOW_TF32 = True

# ---------------------------------------------------------------------------
# 3. PREFERRED BLAS LIBRARY
#
# PyTorch can use "cublas" (legacy) or "cublaslt" (newer, more options).
# cuBLASLt has more kernel candidates and supports epilogue fusion.
# On AMD: hipBLAS vs hipBLASLt (same distinction).
#
# Options: "default", "cublas", "cublaslt"
# ---------------------------------------------------------------------------
PREFERRED_BLAS = "cublas"

# ---------------------------------------------------------------------------
# 4. DIMENSION PADDING
#
# SAM3's key non-standard dimensions:
#   N=4736 (ViT MLP, not power of 2, not multiple of 128)
#   N=36288 (tracker cross-attn, = 7*5184)
#   M=5184 (= 72*72, not power of 2)
#   M=576 (= 24*24, nice)
#
# Padding to a multiple of 64/128/256 may enable better-aligned kernels.
# Tradeoff: padding wastes compute but may unlock faster tiled kernels.
#
# PAD_TO_MULTIPLE: 0 = no padding, or 64/128/256
#
# AMD relevance: hipBLASLt tile sizes are 64/128/256 aligned.  If padding
# helps on cuBLASLt, it will help even more on hipBLASLt where tile
# flexibility is more limited.
# ---------------------------------------------------------------------------
PAD_TO_MULTIPLE = 0

# ---------------------------------------------------------------------------
# 5. WORKSPACE SIZE (MB)
#
# Larger workspace enables more kernel algorithm choices.
# Both cuBLASLt and hipBLASLt use workspace for split-K reduction,
# staging buffers, and alternative algorithms.
# ---------------------------------------------------------------------------
WORKSPACE_MB = 128

# ---------------------------------------------------------------------------
# 6. DTYPE
#
# float16 vs bfloat16.  Different kernels, different performance.
# SAM3 uses float16 by default, but bfloat16 may have better-tuned kernels
# on some hardware (especially AMD MI300 which emphasizes BF16).
#
# AMD relevance: MI300X is heavily optimized for BF16.  If BF16 is faster
# than FP16 for these shapes, that's directly useful for AMD.
# ---------------------------------------------------------------------------
DTYPE = "bfloat16"

# ---------------------------------------------------------------------------
# 7. LAYOUT / TRANSPOSE STRATEGY
#
# Standard GEMM: C = A @ B where A is (M,K), B is (K,N)
# But cuBLASLt/hipBLASLt can be faster with transposed inputs:
#   C = A @ B^T  (B stored as (N,K) transposed)
#
# For some shapes, pre-transposing B and calling the NT kernel is faster
# than the NN kernel, because the NT kernel has better memory access patterns.
#
# Per-shape transpose override.  True = pre-transpose B.
#
# AMD relevance: hipBLASLt has separate kernels for NN/NT/TN/TT layouts.
# Knowing which layout is faster per shape guides hipBLASLt tuning.
# ---------------------------------------------------------------------------
TRANSPOSE_B_MAP = {
    # (M, N, K) -> True to use B^T layout
    # Shapes where K is small and N is huge may benefit from NT
    (5184, 36288,  256): False,   # trk_xattn_qk
    (5184,   256, 36288): False,  # trk_xattn_v
    (5184,  4736, 1024): False,   # vit_g_mlp_up
    (5184,  1024, 4736): False,   # vit_g_mlp_down
    (5184,  3072, 1024): False,   # vit_g_qkv
    (5184,  5184,  256): False,   # trk_sattn_qk
    (5184,   256, 5184): False,   # trk_sattn_v
    (5184,  1024, 1024): False,   # vit_g_out
    (576,   4736, 1024): False,   # vit_w_mlp_up
    (576,   1024, 4736): False,   # vit_w_mlp_down
    (576,   3072, 1024): False,   # vit_w_qkv
    (5184,  2048,  256): False,   # det_enc_ffn_up
    (5184,   256, 2048): False,   # det_enc_ffn_down
    (400,   5184,  256): False,   # det_xattn_qk
    (32,    4096, 1024): False,   # txt_mlp_up
    (32,    1024, 4096): False,   # txt_mlp_down
}

# ---------------------------------------------------------------------------
# 8. BATCHED GEMM FOR WINDOWED SHAPES
#
# SAM3's ViT has 9 windows per image.  Instead of 9 individual GEMMs of
# shape (576, N, K), we can batch them as one (9, 576, N, K) batched GEMM.
# torch.bmm / cuBLASLt batched GEMM may be faster due to better GPU
# utilization (more work per kernel launch).
#
# AMD relevance: hipBLASLt grouped GEMM is a key optimization surface.
# Knowing the speedup from batching directly informs hipBLASLt tuning.
# ---------------------------------------------------------------------------
BATCH_WINDOW_GEMMS = True
WINDOW_BATCH_SIZE = 6

# ---------------------------------------------------------------------------
# 9. SPLIT-K CONFIGURATION
#
# Split-K decomposes K reduction across workgroups. Helps when K is large
# relative to M*N (more parallelism in reduction dimension).
#
# SAM3 analysis:
#   K=256 shapes: DON'T split (already memory-bound)
#   K=36288 (trk_xattn_v): CANDIDATE for split-K
#   K=4736 with small M=576: CANDIDATE
#   K=4096 with M=32: CANDIDATE
# ---------------------------------------------------------------------------
SPLIT_K_MAP = {
    (5184,   256, 36288): 1,  # trk_xattn_v: K=36288
    (576,   1024, 4736):  1,  # vit_w_mlp_down: K=4736, small M
    (32,    1024, 4096):  1,  # txt_mlp_down: K=4096, tiny M
}


# =========================================================================
# FUNCTIONS — called by benchmark.py
# =========================================================================

def apply_global_settings():
    """Apply global PyTorch/BLAS settings from config."""
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    if PREFERRED_BLAS != "default":
        try:
            torch.backends.cuda.preferred_blas_library(PREFERRED_BLAS)
        except Exception:
            pass  # Older PyTorch versions don't support this


def get_dtype():
    """Return the torch dtype to use."""
    if DTYPE == "bfloat16":
        return torch.bfloat16
    return torch.float16


def should_transpose_b(M, N, K):
    """Check if B should be pre-transposed for this shape."""
    return TRANSPOSE_B_MAP.get((M, N, K), False)


def should_batch_windows(M, N, K):
    """Check if this is a windowed shape that should be batched."""
    if not BATCH_WINDOW_GEMMS:
        return False
    # Windowed shapes have M=576
    return M == 576


def get_split_k(M, N, K):
    """Return split-K factor for a shape."""
    return SPLIT_K_MAP.get((M, N, K), 1)


def get_workspace_bytes():
    """Return workspace size in bytes."""
    return WORKSPACE_MB * 1024 * 1024


def pad_dim(dim):
    """Pad a dimension to the nearest multiple of PAD_TO_MULTIPLE."""
    if PAD_TO_MULTIPLE <= 0:
        return dim
    return ((dim + PAD_TO_MULTIPLE - 1) // PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE


def get_padded_shape(M, N, K):
    """Return padded (M, N, K) based on padding config."""
    if PAD_TO_MULTIPLE <= 0:
        return M, N, K
    return pad_dim(M), pad_dim(N), pad_dim(K)
