#!/usr/bin/env python3
"""
Automated search script for autoresearch-cublas-sam3.

Instead of relying on an LLM agent to edit tune_config.py, this script
systematically explores the tuning knobs and records every experiment.

It modifies tune_config.py, runs verify + benchmark, and keeps/reverts
exactly like the Karpathy autoresearch loop.

Usage:
    python3 search.py                  # Run all experiments
    python3 search.py --dry-run        # Show what would be tried
    python3 search.py --only padding   # Only test padding
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


TUNE_CONFIG_PATH = Path("tune_config.py")
BEST_SCORE_FILE = Path(".best_score")
ITERATION_LOG = Path("iteration_log.jsonl")

# All experiments to try, in order.  Each is:
#   (description, {param: value, ...})
# The search modifies tune_config.py for each experiment, runs the
# verify+benchmark loop, and keeps only improvements.

EXPERIMENTS = [
    # ---- Workspace size ----
    ("workspace 64 MB",     {"WORKSPACE_MB": 64}),
    ("workspace 128 MB",    {"WORKSPACE_MB": 128}),
    ("workspace 256 MB",    {"WORKSPACE_MB": 256}),
    ("workspace 512 MB",    {"WORKSPACE_MB": 512}),

    # ---- BLAS library ----
    ("blas=cublas (legacy)", {"PREFERRED_BLAS": '"cublas"'}),
    ("blas=cublaslt",       {"PREFERRED_BLAS": '"cublaslt"'}),

    # ---- Precision ----
    ("precision=medium (TF32 throughout)",  {"MATMUL_PRECISION": '"medium"'}),
    ("precision=highest (pure FP32 accum)", {"MATMUL_PRECISION": '"highest"'}),
    ("allow_tf32=False",    {"ALLOW_TF32": "False"}),

    # ---- dtype ----
    ("dtype=bfloat16",      {"DTYPE": '"bfloat16"'}),

    # ---- Padding ----
    ("pad to 64",           {"PAD_TO_MULTIPLE": 64}),
    ("pad to 128",          {"PAD_TO_MULTIPLE": 128}),
    ("pad to 256",          {"PAD_TO_MULTIPLE": 256}),

    # ---- Batched window GEMMs ----
    ("batch windows (9x)",  {"BATCH_WINDOW_GEMMS": "True"}),

    # ---- Transpose B for key shapes ----
    ("transpose B: trk_xattn_qk",
     {"TRANSPOSE_B_MAP[(5184, 36288, 256)]": "True"}),
    ("transpose B: vit_g_mlp_up",
     {"TRANSPOSE_B_MAP[(5184, 4736, 1024)]": "True"}),
    ("transpose B: trk_sattn_qk",
     {"TRANSPOSE_B_MAP[(5184, 5184, 256)]": "True"}),

    # ---- Combinations of winners ----
    # These are generated dynamically after individual experiments
]


def read_config():
    """Read current tune_config.py."""
    return TUNE_CONFIG_PATH.read_text()


def write_config(content):
    """Write tune_config.py."""
    TUNE_CONFIG_PATH.write_text(content)


def apply_param(config_text, param, value):
    """Replace a parameter value in tune_config.py text."""
    # Handle dict-key assignments like TRANSPOSE_B_MAP[(5184, 36288, 256)]
    if "[" in param:
        base, key = param.split("[", 1)
        key = key.rstrip("]")
        # Find the dict entry and change its value
        pattern = re.escape(key) + r"\):\s*(True|False|[0-9]+)"
        match = re.search(pattern, config_text)
        if match:
            old = match.group(0)
            new = old[:old.rfind(match.group(1))] + str(value)
            return config_text.replace(old, new, 1)
        return config_text

    # Handle simple assignments: PARAM = value
    pattern = rf"^({re.escape(param)}\s*=\s*)(.+)$"
    return re.sub(pattern, rf"\g<1>{value}", config_text, count=1, flags=re.MULTILINE)


def get_best_score():
    if BEST_SCORE_FILE.exists():
        return float(BEST_SCORE_FILE.read_text().strip())
    return 999999.0


def save_best_score(score):
    BEST_SCORE_FILE.write_text(str(score))


def run_experiment(description, iteration):
    """Run verify + benchmark.  Returns (score, passed)."""
    print(f"\n{'='*72}")
    print(f"  Experiment {iteration}: {description}")
    print(f"  Best so far: {get_best_score():.4f}")
    print(f"{'='*72}")

    # Verify
    print("\n--- Verify ---")
    result = subprocess.run(
        ["python3", "verify.py"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print("  FAILED correctness check")
        if result.stderr:
            print(result.stderr[:500])
        return None, False
    # Print just the result line
    for line in result.stdout.split("\n"):
        if "PASS" in line or "FAIL" in line or "ALL TESTS" in line:
            pass  # skip verbose output
    print("  Correctness OK")

    # Benchmark
    print("\n--- Benchmark ---")
    result = subprocess.run(
        ["python3", "benchmark.py"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print("  FAILED benchmark")
        if result.stderr:
            print(result.stderr[:500])
        return None, False

    # Print benchmark output
    for line in result.stdout.split("\n"):
        if line.strip():
            print(f"  {line}")

    # Extract score
    match = re.search(r"FINAL_SCORE=([-0-9.]+)", result.stdout)
    if not match:
        print("  Could not parse score")
        return None, False

    score = float(match.group(1))
    return score, True


def log_iteration(iteration, description, score, best, kept):
    """Append to iteration log (JSONL)."""
    entry = {
        "iteration": iteration,
        "description": description,
        "score": score,
        "best": best,
        "kept": kept,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(ITERATION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show experiments without running")
    parser.add_argument("--only", type=str, help="Only run experiments matching this keyword")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [(d, p) for d, p in experiments if args.only.lower() in d.lower()]

    if args.dry_run:
        print(f"Would run {len(experiments)} experiments:")
        for i, (desc, params) in enumerate(experiments, 1):
            print(f"  {i:3d}. {desc}")
            for k, v in params.items():
                print(f"       {k} = {v}")
        return

    # Save original config
    original_config = read_config()

    # Run baseline if needed
    if not BEST_SCORE_FILE.exists():
        print("Running baseline...")
        score, ok = run_experiment("baseline", 0)
        if ok and score is not None:
            save_best_score(score)
            log_iteration(0, "baseline", score, score, True)
            subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                           "user.email=auto@research", "add", "tune_config.py"],
                          capture_output=True)
            subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                           "user.email=auto@research", "commit",
                           "-m", f"baseline: score {score:.4f}"],
                          capture_output=True)
        else:
            print("Baseline failed!")
            sys.exit(1)

    # Run experiments
    kept_improvements = []
    for i, (description, params) in enumerate(experiments, 1):
        best_before = get_best_score()

        # Apply parameter changes
        config = read_config()
        for param, value in params.items():
            config = apply_param(config, param, value)
        write_config(config)

        # Run
        try:
            score, ok = run_experiment(description, i)
        except Exception as e:
            print(f"  Exception: {e}")
            score, ok = None, False

        if ok and score is not None and score < best_before:
            # Improvement!
            delta = best_before - score
            pct = (delta / abs(best_before)) * 100 if best_before != 0 else 0
            print(f"\n  >>> IMPROVED by {delta:.4f} ({pct:+.2f}%)")
            print(f"  >>> New best: {score:.4f}")
            save_best_score(score)
            log_iteration(i, description, score, score, True)
            kept_improvements.append((description, score, pct))

            # Commit
            subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                           "user.email=auto@research", "add", "tune_config.py"],
                          capture_output=True)
            subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                           "user.email=auto@research", "commit",
                           "-m", f"{description}: score {score:.4f} (was {best_before:.4f})"],
                          capture_output=True)

            # Update original to include this improvement
            original_config = read_config()
        else:
            if score is not None:
                print(f"\n  >>> No improvement (score {score:.4f} vs best {best_before:.4f})")
            else:
                print(f"\n  >>> Experiment failed")
            log_iteration(i, description, score if score else 999999, best_before, False)
            # Revert
            write_config(original_config)

    # Summary
    print("\n" + "=" * 72)
    print(f"Search complete: {len(experiments)} experiments, "
          f"{len(kept_improvements)} kept improvements")
    print(f"Final best score: {get_best_score():.4f}")
    if kept_improvements:
        print("\nKept improvements:")
        for desc, score, pct in kept_improvements:
            print(f"  {pct:+6.2f}%  {desc} (score {score:.4f})")
    print("=" * 72)
    print("\nRun 'python3 plot_progress.py' to generate the progress chart.")


if __name__ == "__main__":
    main()
