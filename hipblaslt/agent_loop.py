#!/usr/bin/env python3
"""
Autoresearch agent loop for SAM3 GEMM optimization.

Karpathy-style autoresearch: an LLM agent reads benchmark results,
reasons about what to try next, edits tune_config.py, runs the
experiment, and keeps only improvements.

The LLM dynamically designs each experiment — there is no fixed
script. Each iteration, the model sees:
  - The current tune_config.py
  - Per-shape benchmark results (TFLOP/s, shape dimensions)
  - History of what was tried and whether it was kept or reverted
  - The program.md instructions

From this context, the LLM proposes ONE focused change and explains
its hypothesis. This is the core autoresearch idea: the agent learns
from results and invents novel experiments that a scripted search
would never try.

Modes:
  1. LLM agent (default):  Calls an LLM API to dynamically design
     experiments. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.
  2. Scripted fallback:    Pre-programmed experiment list.
     No API key needed, but NOT true autoresearch.

Usage:
    # True autoresearch (LLM designs experiments):
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 agent_loop.py --iterations 100

    # Or with OpenAI-compatible API (local LLMs, etc.):
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=http://localhost:8000/v1
    python3 agent_loop.py --iterations 100 --provider openai --model my-model

    # Scripted fallback (no API key, NOT true autoresearch):
    python3 agent_loop.py --mode scripted --iterations 100
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TUNE_CONFIG = Path("tune_config.py")
LAST_RESULT = Path("last_result.json")
BEST_SCORE_FILE = Path(".best_score")
ITERATION_LOG = Path("iteration_log.jsonl")
PROGRAM_MD = Path("program.md")


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------

def read_config():
    return TUNE_CONFIG.read_text()


def write_config(content):
    TUNE_CONFIG.write_text(content)


def apply_param(config_text, param, value):
    """Replace a parameter value in tune_config.py text."""
    # Handle dict-key assignments like TRANSPOSE_B_MAP[(5184, 36288, 256)]
    if "[" in param:
        base, key = param.split("[", 1)
        key = key.rstrip("]")
        pattern = re.escape(key) + r"\):\s*(True|False|[0-9]+)"
        match = re.search(pattern, config_text)
        if match:
            old = match.group(0)
            new = old[:old.rfind(match.group(1))] + str(value)
            return config_text.replace(old, new, 1)
        return config_text

    # Simple assignment: PARAM = value
    pattern = rf"^({re.escape(param)}\s*=\s*)(.+)$"
    return re.sub(pattern, rf"\g<1>{value}", config_text, count=1, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Score tracking
# ---------------------------------------------------------------------------

def get_best_score():
    if BEST_SCORE_FILE.exists():
        return float(BEST_SCORE_FILE.read_text().strip())
    return None


def save_best_score(score):
    BEST_SCORE_FILE.write_text(str(score))


def load_history():
    if not ITERATION_LOG.exists():
        return []
    entries = []
    for line in ITERATION_LOG.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def log_iteration(iteration, description, score, best, kept):
    entry = {
        "iteration": iteration,
        "description": description,
        "score": score if score is not None else 999999,
        "best": best,
        "kept": kept,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(ITERATION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def git_commit(message):
    subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                   "user.email=auto@research", "add", "tune_config.py"],
                  capture_output=True)
    subprocess.run(["git", "-c", "user.name=autoresearch", "-c",
                   "user.email=auto@research", "commit", "-m", message],
                  capture_output=True)


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run verify + benchmark. Returns (score, output, success)."""
    # Verify
    try:
        result = subprocess.run(
            ["python3", "verify.py"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None, f"Verify FAILED:\n{result.stderr[:500]}", False
    except subprocess.TimeoutExpired:
        return None, "Verify TIMED OUT", False

    # Benchmark
    try:
        result = subprocess.run(
            ["python3", "benchmark.py"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None, f"Benchmark FAILED:\n{result.stderr[:500]}", False
    except subprocess.TimeoutExpired:
        return None, "Benchmark TIMED OUT", False

    output = result.stdout
    match = re.search(r"FINAL_SCORE=([-0-9.]+)", output)
    if not match:
        return None, "Could not parse score", False

    return float(match.group(1)), output, True


# ---------------------------------------------------------------------------
# Scripted fallback (NOT true autoresearch)
# ---------------------------------------------------------------------------

def propose_scripted(iteration, history):
    """Use pre-programmed agent_brain to propose next change.

    This is a scripted search, NOT dynamically designed by an LLM.
    Use --mode llm for true autoresearch.
    """
    from agent_brain import propose_change
    hypothesis, changes = propose_change("last_result.json", history)
    return hypothesis, changes


# ---------------------------------------------------------------------------
# LLM agent (true autoresearch)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ML performance engineer running an autoresearch loop.
Your goal: edit tune_config.py to maximize SAM3 GEMM throughput.

RULES:
1. Each turn, propose EXACTLY ONE focused change to tune_config.py.
2. Output the COMPLETE new tune_config.py file wrapped in ```python ... ```
3. Explain your hypothesis in 1-2 sentences BEFORE the code block.
4. Study the per-shape results carefully. Focus on the WEAKEST shapes.
5. Build on kept improvements — don't undo what worked.
6. Be creative — try things a scripted search wouldn't think of.
7. Learn from the history. If split-K=4 hurt, don't try split-K=8.
   If NT layout helped shape X, consider it for similar shapes.

The benchmark score is negative TFLOP/s (lower = better).
"""


def propose_llm(iteration, history, provider="anthropic", model=None):
    """Use an LLM to dynamically design the next experiment.

    The LLM sees the current config, per-shape benchmark results,
    and experiment history. It reasons about what to try and proposes
    a single focused change. This is the core of autoresearch.
    """
    program = PROGRAM_MD.read_text() if PROGRAM_MD.exists() else ""
    config = read_config()
    last = LAST_RESULT.read_text() if LAST_RESULT.exists() else "{}"
    best = get_best_score()

    # Show more history so the LLM can learn patterns
    recent = history[-15:]
    hist_str = "\n".join(
        f"  {'KEPT' if h.get('kept') else 'reverted'}: {h.get('description','')} "
        f"(score={h.get('score','?')})"
        for h in recent
    )

    user_msg = (
        f"## Iteration {iteration} | Best: {best}\n\n"
        f"## Experiment history (recent {len(recent)} of {len(history)}):\n"
        f"{hist_str}\n\n"
        f"## Current tune_config.py:\n```python\n{config}\n```\n\n"
        f"## Last benchmark results (per-shape):\n```json\n{last}\n```\n\n"
        f"Study the results carefully. Which shapes are weakest? "
        f"What hypothesis would you test next? Propose ONE change."
    )

    system = SYSTEM_PROMPT + "\n\n" + program

    if provider == "anthropic":
        return _call_anthropic(system, user_msg, model or "claude-sonnet-4-20250514")
    else:
        return _call_openai(system, user_msg, model or "gpt-4o",
                           os.environ.get("OPENAI_BASE_URL"))


def _call_anthropic(system, user_msg, model):
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text
    return _parse_llm_response(text)


def _call_openai(system, user_msg, model, base_url=None):
    """Call OpenAI-compatible API (works with local LLMs too)."""
    try:
        import openai
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    text = response.choices[0].message.content
    return _parse_llm_response(text)


def _parse_llm_response(text):
    """Extract hypothesis and code from LLM response."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        hypothesis = text.split("```")[0].strip()[:120]
        return hypothesis, match.group(1)
    return None, None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch agent loop — LLM dynamically designs experiments")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of iterations (0=forever)")
    parser.add_argument("--mode", choices=["llm", "scripted"], default="llm",
                       help="llm = LLM designs experiments (default), "
                            "scripted = pre-programmed search")
    parser.add_argument("--provider", choices=["anthropic", "openai"],
                       default="anthropic",
                       help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default=None,
                       help="Model name (default: claude-sonnet-4-20250514 for anthropic, "
                            "gpt-4o for openai)")
    args = parser.parse_args()

    # Auto-detect: if no API key, fall back to scripted
    if args.mode == "llm":
        has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        if not has_anthropic_key and not has_openai_key:
            print("WARNING: No ANTHROPIC_API_KEY or OPENAI_API_KEY found.")
            print("         Falling back to --mode scripted (NOT true autoresearch).")
            print("         Set an API key for LLM-driven experiment design.\n")
            args.mode = "scripted"
        elif has_openai_key and not has_anthropic_key:
            args.provider = "openai"

    print("=" * 72)
    print("autoresearch-cublas-sam3")
    if args.mode == "llm":
        print(f"  Mode: LLM agent ({args.provider})")
        print(f"  The LLM dynamically designs each experiment.")
    else:
        print(f"  Mode: scripted (pre-programmed search, NOT true autoresearch)")
    print(f"  Iterations: {args.iterations if args.iterations > 0 else 'unlimited'}")
    print("=" * 72)

    # Baseline
    if not BEST_SCORE_FILE.exists():
        print("\n  Running baseline...")
        score, output, ok = run_experiment()
        if ok and score is not None:
            save_best_score(score)
            log_iteration(0, "baseline", score, score, True)
            git_commit(f"baseline: score {score:.4f}")
            print(f"  Baseline: {score:.4f}")
        else:
            print(f"  Baseline failed: {output}")
            sys.exit(1)

    history = load_history()
    iteration = max((h.get("iteration", 0) for h in history), default=0)
    target = iteration + args.iterations if args.iterations > 0 else float("inf")

    while iteration < target:
        iteration += 1
        best_before = get_best_score()

        print(f"\n{'='*72}")
        print(f"  Iteration {iteration}  |  Best: {best_before:.4f}")
        print(f"{'='*72}")

        # Get proposal from agent
        if args.mode == "llm":
            hypothesis, payload = propose_llm(
                iteration, history, args.provider, args.model)
        else:
            hypothesis, payload = propose_scripted(iteration, history)

        if hypothesis is None:
            print("  Agent has no more ideas. Stopping.")
            break

        hypothesis_short = hypothesis[:100].replace("\n", " ")
        print(f"\n  Hypothesis: {hypothesis_short}")

        # Apply change
        original = read_config()
        if args.mode == "llm" and isinstance(payload, str):
            # LLM mode returns full file content
            write_config(payload)
        elif isinstance(payload, dict):
            # Scripted mode returns param changes
            config = read_config()
            for param, value in payload.items():
                config = apply_param(config, param, value)
            write_config(config)
        else:
            print("  Invalid proposal — skipping")
            continue

        # Run experiment
        print("  Running verify + benchmark...")
        score, output, ok = run_experiment()

        if not ok:
            print(f"  FAILED: {output[:200]}")
            write_config(original)
            log_iteration(iteration, hypothesis_short, None, best_before, False)
            history.append({"iteration": iteration, "description": hypothesis_short,
                          "score": None, "best": best_before, "kept": False})
            continue

        # Compare
        if score < best_before:
            delta = best_before - score
            pct = (delta / abs(best_before)) * 100
            print(f"\n  >>> KEPT! {score:.4f} (was {best_before:.4f}, +{pct:.2f}%)")
            save_best_score(score)
            git_commit(f"{hypothesis_short}: {score:.4f} (was {best_before:.4f})")
            log_iteration(iteration, hypothesis_short, score, score, True)
            history.append({"iteration": iteration, "description": hypothesis_short,
                          "score": score, "best": score, "kept": True})
        else:
            print(f"\n  >>> Reverted ({score:.4f} vs best {best_before:.4f})")
            write_config(original)
            log_iteration(iteration, hypothesis_short, score, best_before, False)
            history.append({"iteration": iteration, "description": hypothesis_short,
                          "score": score, "best": best_before, "kept": False})

    # Summary
    kept = sum(1 for h in history if h.get("kept"))
    total = len(history)
    best = get_best_score()
    baseline = next((h["score"] for h in history if h.get("description") == "baseline"), best)
    improvement = ((abs(best) - abs(baseline)) / abs(baseline)) * 100 if baseline else 0

    print(f"\n{'='*72}")
    print(f"  DONE")
    print(f"  Experiments: {total}  |  Kept: {kept}")
    print(f"  Baseline: {baseline:.4f}  |  Best: {best:.4f}  |  Gain: {improvement:+.2f}%")
    print(f"\n  Run: python3 plot_progress.py")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
