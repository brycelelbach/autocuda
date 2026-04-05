#!/usr/bin/env python3
"""
autocuda.py — autonomous CUDA kernel optimizer.

Analogous to karpathy/autoresearch but for GPU kernel performance:
  - Fixed infrastructure  : bench.cu  (never modified)
  - Editable kernel       : kernel.cuh (only thing Claude touches)
  - Fixed measurement     : 15-second nvbench run on 256 MiB per element type
  - Benchmark metrics     : nvbench reports GPU time and global memory bandwidth;
    multiple types are aggregated via --aggregate.
  - Optimisation target   : --metric bandwidth | time chooses which to maximise
    or minimise.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python autocuda.py [--metric bandwidth|time] [--iterations N] [--bench-timeout SEC]
                         [--aggregate min|mean|max]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO         = Path(__file__).parent.resolve()
KERNEL_FILE  = REPO / "kernel.cuh"
RESULTS_FILE = REPO / "results.tsv"
BUILD_DIR    = REPO / "build"
BENCH_BIN    = BUILD_DIR / "bench"
JSON_OUT     = BUILD_DIR / "_bench_result.json"

MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert CUDA performance engineer optimising a GPU kernel, measured
with nvbench on 256 MiB per element type (int8, fp16, fp32, fp64, complex fp64)
with a fixed 15-second time budget per benchmark.

The user message states --metric, --aggregate, and units. Each scalar comes from
nvbench cold summaries (tag nv/cold/...): one benchmark with a type axis over T,
then combined across axis states with --aggregate (defaults: min for bandwidth, max for time).

  bandwidth  —  nv/cold/bw/global/bytes_per_second → GiB/s; higher is better.
  time       —  nv/cold/time/gpu/mean (seconds) → ms; lower is better.

You will receive:
  1. The current kernel.cuh
  2. The full experiment history (timestamp | value | unit | status | description)

Contract — kernel.cuh MUST define:
  • static constexpr int BLOCK_SIZE
  • template<typename T> __global__ void kernel(const T*, T*, std::size_t num_elements)
  • Explicit template instantiations for every T that bench.cu benchmarks
    (int8_t, __half, float, double, cuDoubleComplex)
  • inline int compute_grid_size(std::size_t num_elements)

bench.cu registers one nvbench benchmark with a type axis; for each T it launches e.g.:
  kernel<float><<<compute_grid_size(n), BLOCK_SIZE, 0, stream>>>(src, dst, n)
  with n = 256 MiB / sizeof(T).

Rules:
  • Only modify kernel.cuh.
  • One incremental change per iteration — keep it explainable.
  • Satisfy the contract above; how you implement the kernel body is your choice.
  • Do not pull unrelated dependencies into kernel.cuh.

Infer what to try next from the kernel, --metric, and the experiment history
(rising or falling numbers, failed builds, prior descriptions). Do not rely on a
fixed checklist of tricks — form hypotheses from the actual bottleneck and
validate them one change at a time.

Return ONLY:
  <kernel>
  ... complete new kernel.cuh content ...
  </kernel>
  <description>One-line description of the change</description>
"""

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
    """Run the benchmark; return (metric_value, unit) or (None, None) on failure."""
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(BENCH_BIN), "--json", str(JSON_OUT),
         "--timeout", str(bench_timeout)],
        capture_output=True, text=True,
    )
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print("Benchmark error:\n" + r.stderr[-2000:])
        return None, None

    value, unit = _parse_metric_json(JSON_OUT, metric, aggregate)
    if value is None:
        value, unit = _parse_metric_stdout(r.stdout, metric, aggregate)
    return value, unit


def _aggregate_values(values: list[float], aggregate: str) -> float:
    """Combine per–benchmark-type metric samples into one scalar."""
    if not values:
        return 0.0
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "max":
        return max(values)
    # aggregate == "min"
    return min(values)


def _parse_metric_json(
    json_path: Path, metric: str, aggregate: str
) -> "tuple[float, str] | tuple[None, None]":
    """
    Extract the selected optimisation metric from nvbench JSON output.

    bench.cu registers one benchmark (type axis over T); we collect the metric
    from each non-skipped state and combine with _aggregate_values.

    Summaries used:
      nv/cold/bw/global/bytes_per_second — global memory bandwidth
      nv/cold/time/gpu/mean — mean GPU time in seconds (duration)
    """
    tag_for_metric = {
        "bandwidth": "nv/cold/bw/global/bytes_per_second",
        "time": "nv/cold/time/gpu/mean",
    }
    want = tag_for_metric.get(metric)
    if want is None:
        return None, None

    raw: list[float] = []
    try:
        with open(json_path) as f:
            data = json.load(f)
        for bench in data.get("benchmarks", []):
            for state in bench.get("states", []):
                if state.get("is_skipped"):
                    continue
                for summ in state.get("summaries", []):
                    if summ.get("tag") != want:
                        continue
                    for item in summ.get("data", []):
                        if item.get("name") == "value":
                            raw.append(float(item["value"]))
                            break
    except Exception as exc:
        print(f"JSON parse error: {exc}")
        return None, None

    if not raw:
        return None, None

    if metric == "bandwidth":
        gibs = [(v / (1024 ** 3)) for v in raw]
        return _aggregate_values(gibs, aggregate), "GiB/s"
    if metric == "time":
        ms = [v * 1000.0 for v in raw]
        return _aggregate_values(ms, aggregate), "ms"
    return None, None


def _parse_metric_stdout(
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
            return _aggregate_values(vals, aggregate), "GiB/s"
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
            return _aggregate_values(vals, aggregate), "ms"
        return None, None
    return None, None

# ---------------------------------------------------------------------------
# Results log
# ---------------------------------------------------------------------------
def init_results() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("timestamp\tmetric_value\tunit\tstatus\tdescription\n")


def log_result(value: "float | None", unit: str, status: str, description: str) -> None:
    ts       = datetime.now().isoformat(timespec="seconds")
    val_str  = f"{value:.4f}" if value is not None else "N/A"
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{ts}\t{val_str}\t{unit}\t{status}\t{description}\n")


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
        f"Optimisation target: --metric {metric} — {goal}\n"
        f"nvbench runs int8, fp16, fp32, fp64, and complex fp64; "
        f"reported values use --aggregate {aggregate} across those benchmarks.\n\n"
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
        description="AutoCUDA: autonomous CUDA kernel optimizer"
    )
    ap.add_argument("--iterations", "-n", type=int, default=20,
                    help="number of optimisation iterations (default: 20)")
    ap.add_argument(
        "--metric", "-m",
        choices=("bandwidth", "time"),
        default="bandwidth",
        help="optimisation target: cold global memory BW (GiB/s) or cold mean GPU time (ms)",
    )
    ap.add_argument("--bench-timeout", type=float, default=15.0,
                    help="nvbench measurement timeout in seconds (default: 15)")
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
    print("AutoCUDA  —  autonomous CUDA kernel optimizer")
    print("=" * 60)
    print(f"\nOptimisation target: --metric {args.metric}")
    print(f"Multi-type aggregate: --aggregate {aggregate}")
    print("\nBuilding initial benchmark...")
    if not build(reconfigure=True):
        sys.exit("Initial build failed. Check CMakeLists.txt / bench.cu.")

    # --- baseline measurement ---
    print(f"\nBaseline measurement ({args.bench_timeout}s budget)...")
    baseline_value, unit = run_bench(args.bench_timeout, args.metric, aggregate)
    if baseline_value is None:
        sys.exit("Baseline benchmark failed.\n")

    print(f"\nBaseline: {baseline_value:.4f} {unit}")
    log_result(baseline_value, unit, "baseline", "initial kernel")

    best_value  = baseline_value
    best_kernel = KERNEL_FILE.read_text()

    # --- optimisation loop ---
    for i in range(1, args.iterations + 1):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Iteration {i}/{args.iterations}   best so far: {best_value:.4f} {unit}")
        print(sep)

        response    = ask_claude(client, i, args.metric, unit, aggregate)
        new_kernel  = extract_kernel(response)
        description = extract_description(response)

        if new_kernel is None:
            print("No <kernel> block found in Claude response — skipping.")
            log_result(None, unit, "parse_error", "no <kernel> block in response")
            continue

        print(f"\nTrying: {description}\n")
        KERNEL_FILE.write_text(new_kernel)

        if not cmake_build():
            log_result(None, unit, "build_error", description)
            KERNEL_FILE.write_text(best_kernel)
            print("Build failed — reverted to best known kernel.")
            continue

        value, _ = run_bench(args.bench_timeout, args.metric, aggregate)
        if value is None:
            log_result(None, unit, "runtime_error", description)
            KERNEL_FILE.write_text(best_kernel)
            print("Benchmark failed — reverted.")
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
                  f"({delta} {unit}, {delta_pct:+.1f}%)  — reverted")

        log_result(value, unit, status, description)

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Done.  Best: {best_value:.4f} {unit}")
    print(f"Full results: {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
