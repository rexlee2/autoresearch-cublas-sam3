"""
SAM3 GEMM shape catalog — extracted from facebookresearch/sam3 source code.

Architecture: 848M params
  - Vision Encoder (ViT): 450M, embed_dim=1024, 32 layers, 16 heads
  - Text Encoder: 300M, width=1024, 24 layers, 16 heads
  - Detector (DETR): d_model=256, 6 encoder + 6 decoder layers
  - Tracker: d_model=256, 4 layers

Input: 1008x1008 image, patch_size=14 -> 72x72 = 5184 patches
Window: 24x24 = 576 tokens, 3x3 = 9 windows per image
"""

# -----------------------------------------------------------------------
# Each entry: (M, N, K, label, weight, count, notes)
#
#   M, N, K   = GEMM dimensions
#   label     = human-readable name
#   weight    = importance in the final score (sums to 1.0)
#   count     = how many times this GEMM executes per inference frame
#   notes     = what makes this shape interesting for tuning
#
# Sorted by total FLOP contribution (count * 2*M*N*K), descending.
# -----------------------------------------------------------------------

SAM3_SHAPES = [
    # ===================================================================
    # TIER 1: Highest FLOP — these dominate inference time
    # ===================================================================

    # Tracker cross-attention: current frame tokens attend to 7 memory
    # frames. 5184 queries x (7*5184 = 36288) keys.  Single head, full
    # 256-dim.  This is the single largest GEMM in SAM3.
    #
    # hipBLASLt default heuristic is almost certainly NOT tuned for
    # M=5184, N=36288, K=256 — this is a very unusual shape (large M*N,
    # tiny K).  Split-K is useless here. Need kernels optimized for
    # memory-bound, wide, shallow matmuls.
    (5184, 36288, 256,
     "tracker_cross_attn_qk",    0.12, 4,
     "Unusual: huge M*N, tiny K=256. Memory-bound. Biggest single GEMM."),

    (5184, 256, 36288,
     "tracker_cross_attn_v",     0.12, 4,
     "Transpose of above. K=36288 is very large — compute-bound."),

    # ViT global attention MLP (4 global blocks out of 32 total).
    # Standard ViT MLP but with non-standard mlp_ratio=4.625 -> 4736.
    # 4736 is NOT a power of 2, NOT a multiple of 128.
    # hipBLASLt tile alignment may be suboptimal.
    (5184, 4736, 1024,
     "vit_global_mlp_up",        0.08, 4,
     "N=4736 is awkward (not power-of-2). Tile alignment matters."),

    (5184, 1024, 4736,
     "vit_global_mlp_down",      0.08, 4,
     "K=4736 awkward. May benefit from specific BLOCK_K choices."),

    # ViT global QKV projection (fused Q, K, V into one matmul).
    # N=3072 = 3 * 1024.  Standard but large M.
    (5184, 3072, 1024,
     "vit_global_qkv",           0.06, 4,
     "Fused QKV. N=3072 is 3*1024, reasonable alignment."),

    # ===================================================================
    # TIER 2: Medium FLOP — repeated many times
    # ===================================================================

    # Tracker self-attention: 5184 tokens, single head, full 256-dim.
    (5184, 5184, 256,
     "tracker_self_attn_qk",     0.05, 4,
     "Square-ish but K=256 is small. Memory-bound."),

    (5184, 256, 5184,
     "tracker_self_attn_v",      0.05, 4,
     "Transpose. K=5184 makes this compute-bound."),

    # ViT global output projection.
    (5184, 1024, 1024,
     "vit_global_out_proj",      0.04, 4,
     "Standard square GEMM, large M."),

    # ViT windowed attention MLP — 28 windowed blocks, each processes
    # 9 windows.  M=576 per window, but batched across B*9 windows.
    # Effective batch: (B*9, 576, 4736, 1024).
    # For batch=1: 9 independent (576, 4736, 1024) GEMMs.
    (576, 4736, 1024,
     "vit_window_mlp_up",        0.06, 252,  # 28 blocks * 9 windows
     "Small M=576, repeated 252x. Batched GEMM candidate."),

    (576, 1024, 4736,
     "vit_window_mlp_down",      0.06, 252,
     "Small M=576, repeated 252x. Batched GEMM candidate."),

    # ViT windowed QKV projection.
    (576, 3072, 1024,
     "vit_window_qkv",           0.04, 252,
     "Small M, repeated 252x. Batch these for throughput."),

    # ViT windowed output projection.
    (576, 1024, 1024,
     "vit_window_out_proj",      0.03, 252,
     "Small M, repeated 252x."),

    # ===================================================================
    # TIER 3: Detector — moderate FLOP
    # ===================================================================

    # Detector encoder FFN (6 layers, operates on 5184 feature tokens).
    (5184, 2048, 256,
     "det_encoder_ffn_up",       0.03, 6,
     "N=2048, K=256. Moderate size."),

    (5184, 256, 2048,
     "det_encoder_ffn_down",     0.03, 6,
     "Transpose of above."),

    # Detector decoder (6 layers, 200 queries or 400 with DAC).
    (400, 2048, 256,
     "det_decoder_ffn_up",       0.02, 6,
     "Small M=400. Decode-like, memory-bound."),

    (400, 256, 2048,
     "det_decoder_ffn_down",     0.02, 6,
     "Small M=400."),

    # Detector cross-attention: 400 queries attend to 5184 features.
    (400, 5184, 256,
     "det_cross_attn_qk",        0.02, 6,
     "Asymmetric: M=400, N=5184, K=256."),

    # ===================================================================
    # TIER 4: Text encoder — small but 24 layers
    # ===================================================================

    # Text encoder (24 layers, context_length=32, width=1024).
    (32, 4096, 1024,
     "text_enc_mlp_up",          0.01, 24,
     "Tiny M=32. Extremely memory-bound. Decode-like."),

    (32, 1024, 4096,
     "text_enc_mlp_down",        0.01, 24,
     "Tiny M=32."),

    (32, 3072, 1024,
     "text_enc_qkv",             0.01, 24,
     "Tiny M=32. 24 layers but negligible FLOP."),
]


# -----------------------------------------------------------------------
# Why SAM3 shapes are especially interesting for hipBLASLt tuning:
#
# 1. UNUSUAL ASPECT RATIOS:
#    - (5184, 36288, 256): huge M*N, tiny K. Very rare in LLM workloads.
#    - (576, 4736, 1024):  small M, odd N. Repeated 252 times.
#    - (400, 5184, 256):   small M, large N, tiny K.
#
# 2. NON-STANDARD DIMENSIONS:
#    - N=4736 (not power-of-2, not multiple of 128 or 256)
#    - N=36288 (= 7 * 5184, very large)
#    - N=5184 (= 72*72, not a standard BLAS size)
#    - M=5184, M=576, M=400, M=32 — wide range
#
# 3. MIXED COMPUTE/MEMORY BOUND:
#    - K=256 shapes are memory-bound (need bandwidth-optimized kernels)
#    - K=4736, K=36288 shapes are compute-bound (need FLOP-optimized)
#    - K=1024 shapes are balanced
#
# 4. BATCHING OPPORTUNITIES:
#    - Windowed attention: 252 identical (576, *, 1024) GEMMs per frame
#    - Could use hipBLASLt grouped/batched GEMM instead of looping
#
# 5. DEFAULT HEURISTIC LIKELY SUBOPTIMAL:
#    - hipBLASLt is heavily tuned for LLM shapes (powers of 2, M=1/128/
#      256/512/1024/2048/4096, standard K=4096/8192/11008)
#    - SAM3's shapes (M=5184, N=4736, K=256) are outside that comfort zone
# -----------------------------------------------------------------------
