#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from itertools import count
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO         = Path(__file__).parent.resolve()
KERNEL_FILE  = REPO / "kernel.cuh"
RESULTS_FILE = REPO / "results.csv"
BUILD_DIR    = REPO / "build"
BENCH_BIN    = BUILD_DIR / "bench"
JSON_OUT     = BUILD_DIR / "_bench_result.json"

MODEL = "claude-opus-4-6"

# Must match the element_types list in bench.cu.
NUM_TYPE_VARIANTS = 5

# ---------------------------------------------------------------------------
# System prompt - loaded from the idea skill file next to this script.
# ---------------------------------------------------------------------------
SKILL_FILE = REPO / "cuda-kernel-optimization-idea-skill.md"
SYSTEM_PROMPT = SKILL_FILE.read_text()

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def cmake_configure() -> bool:
    r = subprocess.run(
        ["cmake", "-B", str(BUILD_DIR), "-S", str(REPO),
         "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if r.returncode != 0:
        print("CMake configure failed:\n" + r.stderr[-3000:])
        return False
    return True


def cmake_build() -> bool:
    r = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--parallel"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if r.returncode != 0:
        print("Build failed:\n" + r.stderr[-3000:])
        return False
    return True


def build(reconfigure: bool = False) -> bool:
    if reconfigure or not (BUILD_DIR / "CMakeCache.txt").exists():
        if not cmake_configure():
            return False
    return cmake_build()

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def run_bench(
    bench_timeout: float, metric: str, aggregate: str
) -> "tuple[float, str] | tuple[None, None]":
    """Run the benchmark; return (metric_value, unit) or (None, None) on failure.

    *bench_timeout* is the total wall-time budget for the entire nvbench run.
    nvbench's ``--timeout`` is per-state, so we divide by the number of type
    variants.  A hard subprocess timeout (with 30 s slack) kills the process if
    nvbench somehow overshoots.
    """
    per_state_timeout = bench_timeout / NUM_TYPE_VARIANTS
    hard_timeout = bench_timeout + 30.0

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [str(BENCH_BIN), "--json", str(JSON_OUT),
             "--timeout", str(per_state_timeout)],
            capture_output=True, text=True,
            timeout=hard_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"Benchmark killed: exceeded hard timeout ({hard_timeout:.0f}s)")
        return None, None
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print("Benchmark error:\n" + r.stderr[-2000:])
        return None, None

    value, unit = parse_metric_json(JSON_OUT, metric, aggregate)
    if value is None:
        value, unit = parse_metric_stdout(r.stdout, metric, aggregate)
    return value, unit


def aggregate_values(values: list[float], aggregate: str) -> float:
    """Combine per-benchmark-type metric samples into one scalar."""
    if not values:
        return 0.0
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "max":
        return max(values)
    # aggregate == "min"
    return min(values)


def summary_value(summaries: dict, tag: str) -> "float | None":
    """Extract a single numeric value from a summary dict keyed by tag."""
    summ = summaries.get(tag)
    if summ is None:
        return None
    for item in summ.get("data", []):
        if item.get("name") == "value":
            return float(item["value"])
    return None


def gmem_total_bytes(summaries: dict) -> "float | None":
    """Sum all declared global memory read + write bytes from nv/gmem/* summaries."""
    total = 0
    found = False
    for tag, summ in summaries.items():
        if tag.startswith("nv/gmem/reads/") or tag.startswith("nv/gmem/writes/"):
            for item in summ.get("data", []):
                if item.get("name") == "value":
                    total += int(item["value"])
                    found = True
                    break
    return total if found else None


def parse_metric_json(
    json_path: Path, metric: str, aggregate: str
) -> "tuple[float, str] | tuple[None, None]":
    """
    Extract the selected optimisation metric from nvbench JSON output.

    Uses batch (hot) measurements for lower variance.  Bandwidth and flops
    are derived from ``nv/batch/time/gpu/mean`` combined with the declared
    byte / element counts (``nv/gmem/*`` and ``nv/element_count/*``).
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"JSON parse error: {exc}")
        return None, None

    raw: list[float] = []
    for bench in data.get("benchmarks", []):
        for state in bench.get("states", []):
            if state.get("is_skipped"):
                continue
            summaries = {s["tag"]: s for s in state.get("summaries", [])}

            batch_time_s = summary_value(summaries, "nv/batch/time/gpu/mean")
            if batch_time_s is None or batch_time_s <= 0:
                continue

            if metric == "time":
                raw.append(batch_time_s * 1000.0)
            elif metric == "bandwidth":
                total_bytes = gmem_total_bytes(summaries)
                if not total_bytes:
                    continue
                raw.append(total_bytes / batch_time_s / (1024 ** 3))
            elif metric == "flops":
                elems = summary_value(summaries, "nv/element_count/NumElements")
                if elems is None:
                    continue
                raw.append(elems / batch_time_s / 1e9)

    if not raw:
        return None, None

    units = {"time": "ms", "bandwidth": "GiB/s", "flops": "GFLOP/s"}
    return aggregate_values(raw, aggregate), units.get(metric, "")


def parse_metric_stdout(
    stdout: str, metric: str, aggregate: str
) -> "tuple[float, str] | tuple[None, None]":
    """Fallback: extract metric from nvbench markdown stdout (all benchmark tables)."""
    if metric == "bandwidth":
        vals = []
        for m in re.finditer(
            r"(\d[\d.]*)\s*(GiB/s|GB/s|TB/s)\b", stdout, re.IGNORECASE
        ):
            v = float(m.group(1))
            u = m.group(2).lower()
            if u == "gib/s":
                vals.append(v)
            elif u == "gb/s":
                vals.append(v * (1000 ** 3) / (1024 ** 3))
            elif u == "tb/s":
                vals.append(v * (1000 ** 4) / (1024 ** 3))
        if vals:
            return aggregate_values(vals, aggregate), "GiB/s"
        return None, None
    if metric == "time":
        # Table rows list CPU Time then GPU Time in µs/ms; take the 2nd duration per row.
        vals = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = re.findall(
                r"\|\s*(\d[\d.]*)\s*(ns|us|µs|ms|s)\s*\|", line, re.IGNORECASE
            )
            if len(cells) < 2:
                continue
            val, u = cells[1]
            val = float(val)
            u = u.lower().replace("µ", "u")
            if u == "ns":
                vals.append(val / 1e6)
            elif u == "us":
                vals.append(val / 1000.0)
            elif u == "ms":
                vals.append(val)
            elif u == "s":
                vals.append(val * 1000.0)
        if vals:
            return aggregate_values(vals, aggregate), "ms"
        return None, None
    return None, None

# ---------------------------------------------------------------------------
# Results log
# ---------------------------------------------------------------------------
def init_results() -> None:
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "metric_value", "unit", "status", "description"])


def log_result(value: "float | None", unit: str, status: str, description: str) -> None:
    ts      = datetime.now().isoformat(timespec="seconds")
    val_str = f"{value:.4f}" if value is not None else "N/A"
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, val_str, unit, status, description])


def read_results() -> str:
    return RESULTS_FILE.read_text() if RESULTS_FILE.exists() else "(none)"

# ---------------------------------------------------------------------------
# Claude interaction
# ---------------------------------------------------------------------------
def ask_claude(
    client: anthropic.Anthropic,
    iteration: int,
    metric: str,
    unit: str,
    aggregate: str,
) -> str:
    kernel  = KERNEL_FILE.read_text()
    history = read_results()
    goal = (
        "minimise execution time (lower is better)"
        if metric == "time"
        else f"maximise the metric ({unit}, higher is better)"
    )
    user_msg = (
        f"Optimisation target: --metric {metric} - {goal}\n"
        f"Reported values use --aggregate {aggregate} across benchmark states.\n\n"
        f"Current kernel.cuh:\n```cuda\n{kernel}\n```\n\n"
        f"Experiment history (values are in {unit}):\n```\n{history}\n```\n\n"
        f"This is iteration {iteration}. Propose the next improvement."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def extract_kernel(text: str) -> "str | None":
    m = re.search(r"<kernel>(.*?)</kernel>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_description(text: str) -> str:
    m = re.search(r"<description>(.*?)</description>", text, re.DOTALL)
    return m.group(1).strip() if m else "no description"

# ---------------------------------------------------------------------------
# Metric comparison (time: lower is better; bandwidth: higher is better)
# ---------------------------------------------------------------------------
def is_improvement(metric: str, value: float, best: float) -> bool:
    if metric == "time":
        return value < best
    return value > best


def metric_delta_str(metric: str, value: float, best: float) -> tuple[str, float]:
    """Return (signed delta description, percentage vs best) for logging."""
    if metric == "time":
        delta = best - value
        pct = 100.0 * delta / best if best != 0 else 0.0
        return f"{delta:+.4f}", pct
    delta = value - best
    pct = 100.0 * delta / best if best != 0 else 0.0
    return f"{delta:+.4f}", pct


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="""
AutoCUDA uses an LLM API to iteratively optimise a CUDA kernel for a target metric
(bandwidth, FLOPS, or execution time).  On each iteration it asks the LLOM to propose
an improved version of kernel.cuh, compiles it against the fixed benchmark harness
bench.cu, measures performance with nvbench, and keeps the change only if the metric
improves - otherwise it reverts to the best known kernel.  Experiment history
is logged to a CSV file so the LLM can learn from earlier attempts.

The benchmark is parameterised over several element types; per-type results
are reduced to a single scalar via --aggregate (min, mean, or max).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python autocuda.py [--metric bandwidth|time|flops] [--iterations N|inf]
                       [--bench-timeout SEC] [--aggregate min|mean|max]
"""
    )
    def iterations_type(v: str) -> float:
        if v.lower() == "inf":
            return math.inf
        n = int(v)
        if n < 0:
            raise argparse.ArgumentTypeError("iterations must be non-negative")
        return float(n)

    ap.add_argument("--iterations", "-n", type=iterations_type, default=math.inf,
                    help="number of optimisation iterations, or 'inf' to run forever (default: inf)")
    ap.add_argument(
        "--metric", "-m",
        choices=("bandwidth", "time", "flops"),
        default="bandwidth",
        help="optimisation target: batch bandwidth (GiB/s), batch time (ms), or flops (GFLOP/s)",
    )
    ap.add_argument("--bench-timeout", type=float, default=15.0,
                    help="total wall-time budget for each nvbench run in seconds; "
                    "divided evenly across type variants (default: 15)")
    ap.add_argument(
        "--aggregate", "-a",
        choices=("min", "mean", "max"),
        default=None,
        help="combine metrics from each element-type benchmark: min, mean, or max. "
        "Defaults: min for --metric bandwidth (bottleneck type), "
        "max for --metric time (slowest type).",
    )
    ap.add_argument("--api-key", type=str, default=None,
                    help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: set ANTHROPIC_API_KEY or pass --api-key")

    client = anthropic.Anthropic(api_key=api_key)
    init_results()

    aggregate = args.aggregate
    if aggregate is None:
        aggregate = "max" if args.metric == "time" else "min"

    # --- initial build ---
    print("=" * 60)
    print("AutoCUDA - autonomous CUDA kernel optimizer")
    print("=" * 60)
    print(f"\nOptimisation target: --metric {args.metric}")
    print(f"Multi-type aggregate: --aggregate {aggregate}")
    print("\nBuilding initial benchmark...")
    if not build(reconfigure=True):
        sys.exit("Initial build failed. Check CMakeLists.txt / bench.cu.")

    # --- baseline measurement ---
    per_state = args.bench_timeout / NUM_TYPE_VARIANTS
    print(f"\nBaseline measurement ({args.bench_timeout}s total, "
          f"{per_state:.1f}s per type)...")
    baseline_value, unit = run_bench(args.bench_timeout, args.metric, aggregate)
    if baseline_value is None:
        sys.exit("Baseline benchmark failed.\n")

    print(f"\nBaseline: {baseline_value:.4f} {unit}")
    log_result(baseline_value, unit, "baseline", "initial kernel")

    best_value  = baseline_value
    best_kernel = KERNEL_FILE.read_text()

    # --- optimisation loop ---
    limit = args.iterations
    for i in count(1):
        if i > limit:
            break
        sep = "=" * 60
        print(f"\n{sep}")
        tag = f"{i}" if math.isinf(limit) else f"{i}/{int(limit)}"
        print(f"Iteration {tag}   best so far: {best_value:.4f} {unit}")
        print(sep)

        response    = ask_claude(client, i, args.metric, unit, aggregate)
        new_kernel  = extract_kernel(response)
        description = extract_description(response)

        if new_kernel is None:
            print("No <kernel> block found in Claude response - skipping.")
            log_result(None, unit, "parse_error", "no <kernel> block in response")
            continue

        print(f"\nTrying: {description}\n")
        KERNEL_FILE.write_text(new_kernel)

        if not cmake_build():
            log_result(None, unit, "build_error", description)
            KERNEL_FILE.write_text(best_kernel)
            print("Build failed - reverted to best known kernel.")
            continue

        value, _ = run_bench(args.bench_timeout, args.metric, aggregate)
        if value is None:
            log_result(None, unit, "runtime_error", description)
            KERNEL_FILE.write_text(best_kernel)
            print("Benchmark failed - reverted.")
            continue

        delta, delta_pct = metric_delta_str(args.metric, value, best_value)

        if is_improvement(args.metric, value, best_value):
            status      = "improved"
            best_value  = value
            best_kernel = new_kernel
            sign = "faster" if args.metric == "time" else "higher"
            print(f"\n  IMPROVED  {value:.4f} {unit}  "
                  f"({delta} {unit}, {delta_pct:+.1f}% {sign})")
        else:
            status = "regressed"
            KERNEL_FILE.write_text(best_kernel)
            print(f"\n  regressed  {value:.4f} {unit}  "
                  f"({delta} {unit}, {delta_pct:+.1f}%) - reverted")

        log_result(value, unit, status, description)

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Done.  Best: {best_value:.4f} {unit}")
    print(f"Full results: {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
