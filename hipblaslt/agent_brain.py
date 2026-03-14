"""
Local agent brain for autoresearch — replaces the Anthropic API call.

Analyzes benchmark results, identifies weak shapes, forms hypotheses,
and proposes targeted changes to tune_config.py.

This mimics what an LLM agent would do: read results, reason about
what might help, and output a single focused change.

Experiment budget: ~120+ unique experiments across phases:
  Phase 1: Single-parameter sweeps (~20 experiments)
  Phase 2: Split-K and transpose exploration (~30 experiments)
  Phase 3: Combinations of winning changes (~30 experiments)
  Phase 4: Fine-grained search around winners (~20 experiments)
  Phase 5: Exotic / adversarial experiments (~20 experiments)
"""

import json
import random
from pathlib import Path


def analyze_results(last_result: dict) -> dict:
    """Analyze benchmark results to find optimization opportunities."""
    results = last_result.get("results", [])
    config = last_result.get("config", {})

    if not results:
        return {"weakest": [], "strongest": [], "avg_tflops": 0}

    # Sort by TFLOP/s
    by_perf = sorted(results, key=lambda r: r.get("tflops", 0))
    avg = sum(r["tflops"] for r in results) / len(results)

    # Find shapes far below average
    weak = [r for r in by_perf if r["tflops"] < avg * 0.8]
    strong = [r for r in by_perf if r["tflops"] > avg * 1.1]

    # Classify shapes
    memory_bound = [r for r in results if r["K"] <= 256]
    compute_bound = [r for r in results if r["K"] >= 4096]
    misaligned = [r for r in results if r["N"] % 128 != 0 and r["N"] > 128]
    small_m = [r for r in results if r["M"] <= 64]
    windowed = [r for r in results if r["M"] == 576]
    large_m = [r for r in results if r["M"] >= 5184]
    tracker = [r for r in results if r["N"] >= 36288 or r["K"] >= 36288]

    return {
        "weakest": weak[:5],
        "strongest": strong[:3],
        "avg_tflops": avg,
        "memory_bound": memory_bound,
        "compute_bound": compute_bound,
        "misaligned": misaligned,
        "small_m": small_m,
        "windowed": windowed,
        "large_m": large_m,
        "tracker": tracker,
        "config": config,
        "all_results": results,
    }


def generate_experiments(analysis: dict, history: list) -> list:
    """
    Generate a prioritized list of experiments based on analysis.

    Each experiment is: (hypothesis, changes_dict)
    where changes_dict maps config lines to new values.

    Organized in phases so early iterations do broad sweeps,
    later iterations do targeted refinements.
    """
    experiments = []
    config = analysis.get("config", {})
    results = analysis.get("all_results", [])
    num_completed = len(history)

    # =====================================================================
    # PHASE 1: Single-parameter sweeps (iterations 1-20)
    # =====================================================================

    # --- DTYPE ---
    if config.get("dtype") == "float16":
        experiments.append((
            "Switch to bfloat16 — BF16 kernels are often better tuned for "
            "non-standard shapes because BF16 was added later with newer kernels",
            {"DTYPE": '"bfloat16"'}
        ))
    elif config.get("dtype") == "bfloat16":
        experiments.append((
            "Switch to float16 — check if FP16 is faster for current config",
            {"DTYPE": '"float16"'}
        ))

    # --- WORKSPACE ---
    ws = config.get("workspace_mb", 32)
    for new_ws in [64, 128, 256, 512]:
        if new_ws != ws:
            experiments.append((
                f"Increase workspace to {new_ws}MB — more workspace enables "
                f"more algorithm choices in cuBLASLt/hipBLASLt",
                {"WORKSPACE_MB": new_ws}
            ))

    # --- PADDING ---
    pad = config.get("pad_to_multiple", 0)
    if pad == 0:
        for p in [64, 128, 256]:
            experiments.append((
                f"Pad dimensions to multiples of {p} — SAM3's N=4736 and "
                f"M=5184 don't align to standard tile sizes. Padding to {p} "
                f"trades extra FLOPs for better kernel selection",
                {"PAD_TO_MULTIPLE": p}
            ))
    else:
        experiments.append((
            "Remove padding — check if the padding overhead hurts more than "
            "the alignment helps",
            {"PAD_TO_MULTIPLE": 0}
        ))

    # --- BATCH WINDOWS ---
    if not config.get("batch_windows", False):
        experiments.append((
            "Enable batched window GEMMs — instead of 9 separate (576,N,K) "
            "calls, batch them into one (9,576,N,K) call for better GPU "
            "utilization and reduced kernel launch overhead",
            {"BATCH_WINDOW_GEMMS": "True"}
        ))

    # --- PRECISION ---
    prec = config.get("matmul_precision", "high")
    if prec != "medium":
        experiments.append((
            "Set precision=medium (full TF32) — fastest accumulation mode, "
            "may change kernel selection to prefer speed over precision",
            {"MATMUL_PRECISION": '"medium"'}
        ))
    if prec != "highest":
        experiments.append((
            "Set precision=highest (pure FP32 accum) — different kernel "
            "codepath, sometimes paradoxically faster for certain shapes",
            {"MATMUL_PRECISION": '"highest"'}
        ))

    # --- BLAS LIBRARY ---
    blas = config.get("preferred_blas", "cublaslt")
    if blas != "cublas":
        experiments.append((
            "Switch to legacy cublas — cuBLAS sometimes has better-tuned "
            "kernels for unusual shapes that cuBLASLt doesn't handle well",
            {"PREFERRED_BLAS": '"cublas"'}
        ))

    # --- TF32 toggle ---
    tf32 = config.get("allow_tf32", True)
    if tf32:
        experiments.append((
            "Disable TF32 — forces different kernel selection, may unlock "
            "kernels optimized for pure FP16/BF16 without TF32 fast path",
            {"ALLOW_TF32": "False"}
        ))
    else:
        experiments.append((
            "Enable TF32 — TF32 fast path may speed up accumulation",
            {"ALLOW_TF32": "True"}
        ))

    # =====================================================================
    # PHASE 2: Split-K and transpose exploration (iterations 20-50)
    # =====================================================================

    # --- TRANSPOSE B for all shapes with small K and large N ---
    all_shapes = [(r["M"], r["N"], r["K"], r.get("label", "")) for r in results]
    for M, N, K, label in all_shapes:
        key = f"({M}, {N}, {K})"
        # Small K, large N: NT layout may give better memory access
        if K <= 512 and N > 1000:
            experiments.append((
                f"Transpose B for {label} {key} — K={K} is small, N={N} is "
                f"large, NT layout may improve memory coalescing",
                {f"TRANSPOSE_B_MAP[{key}]": "True"}
            ))
        # Large K, small N: transpose may help read patterns
        if K >= 2048 and N <= 1024:
            experiments.append((
                f"Transpose B for {label} {key} — K={K} is large, N={N} is "
                f"small, NT layout changes memory access pattern",
                {f"TRANSPOSE_B_MAP[{key}]": "True"}
            ))

    # --- SPLIT-K for all large-K shapes ---
    for r in results:
        M, N, K = r["M"], r["N"], r["K"]
        label = r.get("label", "")
        key = f"({M}, {N}, {K})"
        if K >= 1024:
            for sk in [2, 4, 8]:
                # Skip extreme split-K for shapes where it's unlikely to help
                if sk >= 8 and K < 4096:
                    continue
                experiments.append((
                    f"Split-K={sk} for {label} {key} — K={K} is large, "
                    f"splitting across {sk} workgroups adds parallelism",
                    {f"SPLIT_K_MAP[{key}]": sk}
                ))
        # Even for smaller K, split-K=2 can help if M is small
        if K >= 256 and M <= 576 and K not in (256,):
            for sk in [2, 3]:
                experiments.append((
                    f"Split-K={sk} for small-M {label} {key} — M={M} limits "
                    f"parallelism, split-K={sk} may help fill GPU",
                    {f"SPLIT_K_MAP[{key}]": sk}
                ))

    # =====================================================================
    # PHASE 3: Combinations of winning changes (iterations 50-80)
    # =====================================================================

    # Look at what worked in history and combine winning changes
    kept_changes = []
    for h in history:
        if h.get("kept") and h.get("description", "") != "baseline":
            kept_changes.append(h)

    # --- Dtype + workspace combos ---
    for dtype_val in ['"float16"', '"bfloat16"']:
        for ws_val in [64, 128, 256]:
            current_dtype = f'"{config.get("dtype", "bfloat16")}"'
            current_ws = config.get("workspace_mb", 32)
            if dtype_val != current_dtype or ws_val != current_ws:
                changes = {}
                parts = []
                if dtype_val != current_dtype:
                    changes["DTYPE"] = dtype_val
                    parts.append(f"dtype={dtype_val}")
                if ws_val != current_ws:
                    changes["WORKSPACE_MB"] = ws_val
                    parts.append(f"workspace={ws_val}MB")
                if len(changes) >= 2:
                    experiments.append((
                        f"Combo: {' + '.join(parts)} — test if these "
                        f"changes are synergistic",
                        changes
                    ))

    # --- Precision + TF32 combos ---
    for prec_val in ['"medium"', '"high"', '"highest"']:
        for tf32_val in ["True", "False"]:
            current_prec = f'"{config.get("matmul_precision", "high")}"'
            current_tf32 = str(config.get("allow_tf32", True))
            if prec_val != current_prec or tf32_val != current_tf32:
                changes = {}
                parts = []
                if prec_val != current_prec:
                    changes["MATMUL_PRECISION"] = prec_val
                    parts.append(f"precision={prec_val}")
                if tf32_val != current_tf32:
                    changes["ALLOW_TF32"] = tf32_val
                    parts.append(f"tf32={tf32_val}")
                if len(changes) >= 2:
                    experiments.append((
                        f"Combo: {' + '.join(parts)} — different precision "
                        f"modes interact with TF32 kernel selection",
                        changes
                    ))

    # --- Batch windows + workspace combos ---
    batch = config.get("batch_windows", False)
    for batch_val in ["True", "False"]:
        for ws_val in [64, 128, 256]:
            current_batch = str(batch)
            current_ws = config.get("workspace_mb", 32)
            if batch_val != current_batch or ws_val != current_ws:
                changes = {}
                parts = []
                if batch_val != current_batch:
                    changes["BATCH_WINDOW_GEMMS"] = batch_val
                    parts.append(f"batch_windows={batch_val}")
                if ws_val != current_ws:
                    changes["WORKSPACE_MB"] = ws_val
                    parts.append(f"workspace={ws_val}MB")
                if len(changes) >= 2:
                    experiments.append((
                        f"Combo: {' + '.join(parts)} — batched GEMMs may need "
                        f"different workspace size",
                        changes
                    ))

    # --- Multiple split-K settings at once ---
    # Try setting split-K for multiple shapes simultaneously
    compute_shapes = [(r["M"], r["N"], r["K"], r.get("label", ""))
                      for r in results if r["K"] >= 4096]
    if len(compute_shapes) >= 2:
        for sk in [2, 4]:
            changes = {}
            labels = []
            for M, N, K, label in compute_shapes:
                key = f"({M}, {N}, {K})"
                changes[f"SPLIT_K_MAP[{key}]"] = sk
                labels.append(label)
            experiments.append((
                f"Split-K={sk} for ALL large-K shapes ({', '.join(labels[:3])}) "
                f"— coordinated split-K may improve overall throughput",
                changes
            ))

    # --- Multiple transpose at once ---
    small_k_large_n = [(r["M"], r["N"], r["K"], r.get("label", ""))
                       for r in results if r["K"] <= 256 and r["N"] >= 2048]
    if len(small_k_large_n) >= 2:
        changes = {}
        labels = []
        for M, N, K, label in small_k_large_n:
            key = f"({M}, {N}, {K})"
            changes[f"TRANSPOSE_B_MAP[{key}]"] = "True"
            labels.append(label)
        experiments.append((
            f"Transpose B for ALL memory-bound shapes ({', '.join(labels[:3])}) "
            f"— coordinated NT layout for small-K shapes",
            changes
        ))

    # =====================================================================
    # PHASE 4: Fine-grained workspace search (iterations 80-90)
    # =====================================================================

    for ws_val in [96, 160, 192, 384, 48]:
        if ws_val != config.get("workspace_mb", 32):
            experiments.append((
                f"Fine-tune workspace to {ws_val}MB — searching between "
                f"coarse grid points for optimal workspace size",
                {"WORKSPACE_MB": ws_val}
            ))

    # =====================================================================
    # PHASE 5: Exotic / adversarial experiments (iterations 90+)
    # =====================================================================

    # --- Disable batch windows (if currently enabled) ---
    if config.get("batch_windows", False):
        experiments.append((
            "Disable batched window GEMMs — individual launches may be "
            "faster if the batched kernel path is suboptimal",
            {"BATCH_WINDOW_GEMMS": "False"}
        ))

    # --- Swap BLAS with other knobs ---
    if blas != "cublas":
        for prec_val in ['"medium"', '"highest"']:
            experiments.append((
                f"Legacy cublas + precision={prec_val} — cublas with "
                f"different precision may find better kernel paths",
                {"PREFERRED_BLAS": '"cublas"', "MATMUL_PRECISION": prec_val}
            ))

    # --- Full reversal: baseline settings with one kept tweak ---
    for h in kept_changes[:5]:
        desc = h.get("description", "")
        if "split-k" in desc.lower() or "Split-K" in desc:
            experiments.append((
                f"Revert all except split-K wins — isolate split-K "
                f"benefit from other accumulated changes",
                {"DTYPE": '"bfloat16"', "WORKSPACE_MB": 32,
                 "MATMUL_PRECISION": '"high"', "ALLOW_TF32": "True",
                 "PAD_TO_MULTIPLE": 0}
            ))
            break

    # --- Window batch size variations ---
    for bs in [4, 6, 12, 16]:
        experiments.append((
            f"Window batch size {bs} — instead of 9 (3x3 windows), try "
            f"batch={bs} to test if different grouping helps throughput",
            {"WINDOW_BATCH_SIZE": bs}
        ))

    # --- Pad only specific dimensions ---
    for p in [64, 128]:
        # These change PAD_TO_MULTIPLE but also suggest the idea
        experiments.append((
            f"Pad to {p} + workspace 256MB — combine alignment with extra "
            f"workspace to offset padding overhead",
            {"PAD_TO_MULTIPLE": p, "WORKSPACE_MB": 256}
        ))

    # --- dtype + split-K combos ---
    for dtype_val in ['"float16"', '"bfloat16"']:
        current_dtype = f'"{config.get("dtype", "bfloat16")}"'
        if dtype_val != current_dtype:
            for r in results:
                if r["K"] >= 4096 and r["M"] <= 576:
                    key = f"({r['M']}, {r['N']}, {r['K']})"
                    label = r.get("label", "")
                    for sk in [2, 4]:
                        experiments.append((
                            f"dtype={dtype_val} + split-K={sk} for {label} "
                            f"— different dtype may prefer different split-K",
                            {"DTYPE": dtype_val, f"SPLIT_K_MAP[{key}]": sk}
                        ))

    # --- cublas + transpose combos ---
    for r in results:
        if r["K"] <= 256 and r["N"] >= 5000:
            key = f"({r['M']}, {r['N']}, {r['K']})"
            label = r.get("label", "")
            experiments.append((
                f"cublas + transpose B for {label} — legacy cublas may have "
                f"better NT kernels for this shape",
                {"PREFERRED_BLAS": '"cublas"',
                 f"TRANSPOSE_B_MAP[{key}]": "True"}
            ))

    # --- Random combo perturbations (for late-stage exploration) ---
    random.seed(42 + num_completed)
    for i in range(10):
        changes = {}
        parts = []

        # Pick 2-3 random knobs to tweak
        knobs = random.sample([
            ("DTYPE", ['"float16"', '"bfloat16"']),
            ("WORKSPACE_MB", [32, 64, 128, 256]),
            ("MATMUL_PRECISION", ['"medium"', '"high"', '"highest"']),
            ("ALLOW_TF32", ["True", "False"]),
            ("BATCH_WINDOW_GEMMS", ["True", "False"]),
            ("PAD_TO_MULTIPLE", [0, 64, 128]),
        ], k=random.randint(2, 3))

        for param, values in knobs:
            val = random.choice(values)
            changes[param] = val
            parts.append(f"{param}={val}")

        experiments.append((
            f"Random combo #{i+1}: {' + '.join(parts)} — late-stage "
            f"exploration of parameter interactions",
            changes
        ))

    # =====================================================================
    # DEDUP: filter out already-tried experiments
    # =====================================================================
    tried_sigs = set()
    for h in history:
        desc = h.get("description", "")
        tried_sigs.add(desc[:80])

    filtered = []
    seen_sigs = set()
    for hyp, changes in experiments:
        sig = str(sorted(changes.items()))
        if sig not in seen_sigs and hyp[:80] not in tried_sigs:
            seen_sigs.add(sig)
            filtered.append((hyp, changes))

    return filtered


def propose_change(last_result_path: str = "last_result.json",
                   history: list = None) -> tuple:
    """
    Propose the next change to tune_config.py.

    Returns: (hypothesis: str, changes: dict) or (None, None) if no more ideas.
    """
    if history is None:
        history = []

    # Load results
    try:
        with open(last_result_path) as f:
            last_result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        last_result = {}

    analysis = analyze_results(last_result)
    experiments = generate_experiments(analysis, history)

    if not experiments:
        return None, None

    # Pick the first untried experiment (prioritized order)
    return experiments[0]
