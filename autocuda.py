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
from datetime import date, datetime
from itertools import count
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO         = Path(__file__).parent.resolve()
RESULTS_FILE = REPO / "results.csv"
BUILD_DIR    = REPO / "build"
JSON_OUT     = BUILD_DIR / "_bench_result.json"

BASE_URL = "https://inference-api.nvidia.com/v1/"
DEFAULT_MODEL = "aws/anthropic/bedrock-claude-opus-4-6"

# ---------------------------------------------------------------------------
# Per-kernel configuration.  Set by configure_kernel() after arg parsing.
# ---------------------------------------------------------------------------
KERNEL_CONFIGS = {
    "memcpy":  {"dir": "kernels/memcpy",  "target": "bench_memcpy",  "num_variants": 5},
    "stencil": {"dir": "kernels/stencil", "target": "bench_stencil", "num_variants": 2},
    "matmul":  {"dir": "kernels/matmul",  "target": "bench_matmul",  "num_variants": 3},
    "sigmoid": {"dir": "kernels/sigmoid", "target": "bench_sigmoid", "num_variants": 3},
}

KERNEL_FILE: Path
BENCH_BIN: Path
BENCH_TARGET: str
NUM_TYPE_VARIANTS: int


def configure_kernel(name: str) -> None:
    global KERNEL_FILE, BENCH_BIN, BENCH_TARGET, NUM_TYPE_VARIANTS
    cfg = KERNEL_CONFIGS[name]
    KERNEL_FILE       = REPO / cfg["dir"] / "kernel.cuh"
    BENCH_BIN         = BUILD_DIR / cfg["target"]
    BENCH_TARGET      = cfg["target"]
    NUM_TYPE_VARIANTS = cfg["num_variants"]

# ---------------------------------------------------------------------------
# System prompt - loaded from the idea skill file next to this script.
# ---------------------------------------------------------------------------
SKILL_FILE = REPO / "cuda-kernel-optimization-idea-skill.md"
SYSTEM_PROMPT = SKILL_FILE.read_text()

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True, cwd=str(REPO), check=check,
    )


def git_setup_branch(tag: str, continue_existing: bool) -> None:
    """Create or check out ``autocuda/<tag>``.

    When *continue_existing* is ``True`` and the branch already exists, it is
    checked out so the run can resume.  When ``False``, an existing branch is
    treated as an error.
    """
    branch = f"autocuda/{tag}"
    probe = _git("rev-parse", "--verify", branch, check=False)
    exists = probe.returncode == 0

    if exists and not continue_existing:
        sys.exit(f"Error: branch '{branch}' already exists. "
                 "Pick a different --tag, pass --continue, or delete the branch first.")

    if exists:
        r = _git("checkout", branch)
        if r.returncode != 0:
            sys.exit(f"Failed to check out branch '{branch}':\n{r.stderr}")
        print(f"Continuing on branch: {branch}")
    else:
        if continue_existing:
            sys.exit(f"Error: branch '{branch}' does not exist. "
                     "Remove --continue to start a new run.")
        r = _git("checkout", "-b", branch)
        if r.returncode != 0:
            sys.exit(f"Failed to create branch '{branch}':\n{r.stderr}")
        print(f"Created branch: {branch}")


def git_commit_kernel(description: str, iteration: int) -> None:
    """Stage ``kernel.cuh`` and commit with the iteration description."""
    _git("add", str(KERNEL_FILE.relative_to(REPO)))
    _git("commit", "-m", f"AutoCUDA Iteration {iteration}: {description}", check=False)


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


def cmake_build() -> tuple[bool, str]:
    r = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--parallel",
         "--target", BENCH_TARGET],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if r.returncode != 0:
        err = r.stderr[-3000:]
        print("Build failed:\n" + err)
        return False, err
    return True, ""


def build(reconfigure: bool = False) -> bool:
    if reconfigure or not (BUILD_DIR / "CMakeCache.txt").exists():
        if not cmake_configure():
            return False
    ok, _ = cmake_build()
    return ok

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def run_bench(
    bench_timeout: float, metric: str, aggregate: str
) -> "tuple[float | None, list[tuple[str, float]], str | None, str]":
    """Run the benchmark; return (aggregate_value, per_variant, unit, error_msg).

    *per_variant* is ``[(state_name, metric_value), ...]`` when JSON parsing
    succeeds, or ``[]`` when falling back to stdout parsing.

    On success *error_msg* is empty.  On failure *aggregate_value* and *unit*
    are ``None``, *per_variant* is ``[]``, and *error_msg* describes what went
    wrong.

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
        msg = f"Benchmark killed: exceeded hard timeout ({hard_timeout:.0f}s)"
        print(msg)
        return None, [], None, msg
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode != 0:
        err = (r.stderr[-2000:] + "\n" + r.stdout[-1000:]).strip()
        print("Benchmark error:\n" + err)
        return None, [], None, err

    per_variant, value, unit = parse_metric_json_detailed(JSON_OUT, metric, aggregate)
    if value is None:
        value, unit = parse_metric_stdout(r.stdout, metric, aggregate)
        per_variant = []
    return value, per_variant, unit, ""


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
    """Convenience wrapper: returns (aggregate_value, unit) only."""
    per_variant, agg, unit = parse_metric_json_detailed(json_path, metric, aggregate)
    if agg is None:
        return None, None
    return agg, unit


def parse_metric_json_detailed(
    json_path: Path, metric: str, aggregate: str
) -> "tuple[list[tuple[str, float]], float, str] | tuple[list, None, None]":
    """
    Extract per-variant metric values and their aggregate from nvbench JSON.

    Returns ``(per_variant, aggregate_value, unit)`` on success, where
    *per_variant* is ``[(state_name, value), ...]``.  On failure returns
    ``([], None, None)``.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"JSON parse error: {exc}")
        return [], None, None

    per_variant: list[tuple[str, float]] = []
    for bench in data.get("benchmarks", []):
        for state in bench.get("states", []):
            if state.get("is_skipped"):
                continue
            summaries = {s["tag"]: s for s in state.get("summaries", [])}

            batch_time_s = summary_value(summaries, "nv/batch/time/gpu/mean")
            if batch_time_s is None or batch_time_s <= 0:
                continue

            state_name = state.get("name", "?")
            if metric == "time":
                per_variant.append((state_name, batch_time_s * 1000.0))
            elif metric == "memory-bandwidth":
                total_bytes = gmem_total_bytes(summaries)
                if not total_bytes:
                    continue
                per_variant.append(
                    (state_name, total_bytes / batch_time_s / (1024 ** 3))
                )
            elif metric == "compute-bandwidth":
                elems = summary_value(summaries, "nv/element_count/NumElements")
                if elems is None:
                    continue
                per_variant.append((state_name, elems / batch_time_s / 1e9))

    if not per_variant:
        return [], None, None

    raw = [v for _, v in per_variant]
    units = {"memory-bandwidth": "GiB/s", "compute-bandwidth": "GFLOP/s", "time": "ms"}
    return per_variant, aggregate_values(raw, aggregate), units.get(metric, "")


def parse_metric_stdout(
    stdout: str, metric: str, aggregate: str
) -> "tuple[float, str] | tuple[None, None]":
    """Fallback: extract metric from nvbench markdown stdout (all benchmark tables)."""
    if metric == "memory-bandwidth":
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


def count_existing_iterations() -> int:
    """Count non-baseline rows in results.csv to determine the iteration offset."""
    if not RESULTS_FILE.exists():
        return 0
    with open(RESULTS_FILE) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for row in reader if len(row) >= 4 and row[3] != "baseline")

# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def ask_llm(
    client: OpenAI,
    model: str,
    iteration: int,
    metric: str,
    unit: str,
    aggregate: str,
    failed_kernel: "str | None" = None,
    failure_reason: str = "",
    failure_output: str = "",
    direction: "str | None" = None,
) -> str:
    kernel  = KERNEL_FILE.read_text()
    history = read_results()
    goal = (
        "minimise execution time (lower is better)"
        if metric == "time"
        else f"maximise the metric ({unit}, higher is better)"
    )

    parts: list[str] = []

    if failed_kernel is not None:
        parts.append(
            f"WARNING: Your previous iteration FAILED ({failure_reason}).\n\n"
            f"Error output:\n```\n{failure_output[-3000:]}\n```\n\n"
            f"The failed kernel you proposed:\n```cuda\n{failed_kernel}\n```\n\n"
            "You MUST fix the issues in your next proposal. "
            "kernel.cuh has been reverted to the last known good version shown below.\n\n"
        )

    parts.append(
        f"Optimization target: --metric {metric} - {goal}\n"
        f"Reported values use --aggregate {aggregate} across benchmark states.\n\n"
        f"Current kernel.cuh:\n```cuda\n{kernel}\n```\n\n"
        f"Experiment history (values are in {unit}):\n```\n{history}\n```\n\n"
    )

    if direction:
        parts.append(
            f"IMPORTANT DIRECTION FROM THE USER — you MUST follow this guidance:\n\n"
            f"{direction}\n\n"
        )

    parts.append(f"This is iteration {iteration}. Propose the next improvement.")

    user_msg = "".join(parts)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=65536,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content


def extract_kernel(text: str) -> "str | None":
    m = re.search(r"<kernel>(.*?)</kernel>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_description(text: str) -> str:
    m = re.search(r"<description>(.*?)</description>", text, re.DOTALL)
    return m.group(1).strip() if m else "no description"

# ---------------------------------------------------------------------------
# Metric comparison (time: lower is better; memory-bandwidth/compute-bandwidth: higher is better)
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
# LLM-based accept/reject decision
# ---------------------------------------------------------------------------
DECISION_SYSTEM_PROMPT = """\
You are evaluating whether to accept or reject a proposed change to a CUDA \
kernel that is being iteratively optimized for performance.

## Decision criteria

1. **Performance improvement**: Accept changes that improve the target metric.
2. **Simplicity trade-off**: Accept changes that simplify the code (fewer \
lines, clearer logic, higher-level abstractions) even if they cause a \
minor performance regression.  A small regression is acceptable when the \
code is meaningfully simpler or more maintainable.
3. **Mixed variant results**: If some type variants improve and others regress, \
weigh the overall picture.  A change that substantially improves most variants \
but slightly regresses one is generally worth keeping.
4. **No benefit**: Reject changes that neither improve performance nor simplify \
the code.

## Output format

Return ONLY:
<decision>accept</decision> or <decision>reject</decision>
<reasoning>Brief explanation of your decision (1-3 sentences)</reasoning>
"""


def format_variant_table(
    metric: str,
    unit: str,
    aggregate: str,
    old_per_variant: "list[tuple[str, float]]",
    new_per_variant: "list[tuple[str, float]]",
    old_aggregate: float,
    new_aggregate: float,
) -> str:
    """Build a markdown table comparing per-variant and aggregate metrics."""
    higher_better = metric != "time"
    direction = "higher is better" if higher_better else "lower is better"

    old_map = dict(old_per_variant)
    new_map = dict(new_per_variant)
    all_names = list(dict.fromkeys(
        [n for n, _ in old_per_variant] + [n for n, _ in new_per_variant]
    ))

    lines = [
        f"Metric: {metric} ({unit}, {direction})",
        f"Aggregate method: {aggregate}",
        "",
        "| Variant | Old | New | Delta | % Change |",
        "|---------|----:|----:|------:|---------:|",
    ]
    for name in all_names:
        ov = old_map.get(name)
        nv = new_map.get(name)
        if ov is not None and nv is not None:
            delta = nv - ov
            pct = 100.0 * delta / ov if ov != 0 else 0.0
            lines.append(
                f"| {name} | {ov:.4f} | {nv:.4f} | {delta:+.4f} | {pct:+.1f}% |"
            )
        elif ov is not None:
            lines.append(f"| {name} | {ov:.4f} | N/A | - | - |")
        elif nv is not None:
            lines.append(f"| {name} | N/A | {nv:.4f} | - | - |")

    agg_delta = new_aggregate - old_aggregate
    agg_pct = 100.0 * agg_delta / old_aggregate if old_aggregate != 0 else 0.0
    lines.append(
        f"| **Aggregate ({aggregate})** | **{old_aggregate:.4f}** "
        f"| **{new_aggregate:.4f}** | **{agg_delta:+.4f}** | **{agg_pct:+.1f}%** |"
    )
    return "\n".join(lines)


def ask_llm_decision(
    client: OpenAI,
    model: str,
    metric: str,
    unit: str,
    aggregate: str,
    old_kernel: str,
    new_kernel: str,
    description: str,
    old_per_variant: "list[tuple[str, float]]",
    new_per_variant: "list[tuple[str, float]]",
    old_aggregate: float,
    new_aggregate: float,
) -> "tuple[bool, str]":
    """Ask the LLM whether to accept or reject a kernel change.

    Returns ``(accept, reasoning)``.  Falls back to the hard-coded
    ``is_improvement`` check if the LLM response cannot be parsed.
    """
    table = format_variant_table(
        metric, unit, aggregate,
        old_per_variant, new_per_variant,
        old_aggregate, new_aggregate,
    )

    user_msg = (
        f"## Proposed change\n\n{description}\n\n"
        f"## Current best kernel\n```cuda\n{old_kernel}\n```\n\n"
        f"## Proposed kernel\n```cuda\n{new_kernel}\n```\n\n"
        f"## Performance comparison\n\n{table}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content
    except Exception as exc:
        print(f"LLM decision call failed ({exc}); falling back to metric comparison.")
        accept = is_improvement(metric, new_aggregate, old_aggregate)
        return accept, "fallback: LLM call failed"

    accept, reasoning = extract_decision(text)
    if accept is None:
        print(f"Could not parse LLM decision; falling back to metric comparison.\n"
              f"LLM response: {text[:500]}")
        accept = is_improvement(metric, new_aggregate, old_aggregate)
        reasoning = "fallback: unparseable LLM response"

    return accept, reasoning


def extract_decision(text: str) -> "tuple[bool | None, str]":
    """Parse ``<decision>accept/reject</decision>`` and ``<reasoning>...</reasoning>``."""
    dm = re.search(r"<decision>\s*(accept|reject)\s*</decision>", text, re.IGNORECASE)
    rm = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    reasoning = rm.group(1).strip() if rm else ""
    if dm is None:
        return None, reasoning
    return dm.group(1).lower() == "accept", reasoning


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="""
AutoCUDA uses an LLM API to iteratively optimize a CUDA kernel for a target metric
(memory-bandwidth, compute-bandwidth, or time).  On each iteration it asks the LLM to propose
an improved version of kernel.cuh, compiles it against the fixed benchmark harness
bench.cu, measures performance with nvbench, and then asks the LLM whether to keep or
reject the change based on the full per-variant performance breakdown and code
complexity.  Experiment history is logged to a CSV file so the LLM can learn from
earlier attempts.

The benchmark is parameterised over several variants; per-variant results
are reduced to a single scalar via --aggregate (min, mean, or max).

Usage:
    export NVIDIA_API_KEY_AUTOCUDA=...
    python autocuda.py [--metric memory-bandwidth|compute-bandwidth|time] [--iterations N|inf]
                       [--bench-timeout SEC] [--aggregate min|mean|max] [--continue]
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
                    help="number of optimization iterations, or 'inf' to run forever (default: inf)")
    ap.add_argument(
        "--metric", "-m",
        choices=("memory-bandwidth", "compute-bandwidth", "time"),
        default="memory-bandwidth",
        help="optimization target: memory-bandwidth (GiB/s), compute-bandwidth (GFLOP/s), or time (ms)",
    )
    ap.add_argument("--bench-timeout", type=float, default=15.0,
                    help="total wall-time budget for each nvbench run in seconds; "
                    "divided evenly across type variants (default: 15)")
    ap.add_argument(
        "--aggregate", "-a",
        choices=("min", "mean", "max"),
        default="mean",
        help="how metrics from different benchmark variants should be combined: min, mean, or max "
        "(default: mean).",
    )
    default_tag = date.today().isoformat()
    ap.add_argument("--tag", type=str, default=default_tag,
                    help="run tag for the git branch autocuda/<tag> "
                    f"(default: {default_tag})")
    ap.add_argument("--continue", dest="continue_run", action="store_true",
                    default=False,
                    help="resume on an existing autocuda/<tag> branch instead of creating a new one")
    ap.add_argument("--direction", "-d", type=str, default=None,
                    help="optional guidance message injected into the LLM prompt to steer experimentation")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"model identifier (default: {DEFAULT_MODEL})")
    ap.add_argument("--api-key", type=str, default=None,
                    help="NVIDIA inference API key (overrides NVIDIA_API_KEY_AUTOCUDA env var)")
    ap.add_argument(
        "--kernel", "-k",
        choices=tuple(KERNEL_CONFIGS),
        default="memcpy",
        help="which kernel to optimize: "
             + ", ".join(KERNEL_CONFIGS) + " (default: memcpy)",
    )
    args = ap.parse_args()

    configure_kernel(args.kernel)

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY_AUTOCUDA")
    if not api_key:
        sys.exit("Error: set NVIDIA_API_KEY_AUTOCUDA or pass --api-key")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    init_results()

    aggregate = args.aggregate

    # --- git branch ---
    tag = args.tag
    git_setup_branch(tag, args.continue_run)
    iter_offset = count_existing_iterations() if args.continue_run else 0

    # --- initial build ---
    print("=" * 60)
    print("AutoCUDA - autonomous CUDA kernel optimizer")
    print("=" * 60)
    print(f"\nOptimization target: --metric {args.metric}")
    print(f"Multi-type aggregate: --aggregate {aggregate}")
    if args.continue_run:
        print(f"Resuming from iteration {iter_offset}")
    if args.direction:
        print(f"Direction: {args.direction}")
    print("\nBuilding initial benchmark...")
    if not build(reconfigure=True):
        sys.exit("Initial build failed. Check CMakeLists.txt / bench.cu.")

    # --- baseline measurement ---
    per_state = args.bench_timeout / NUM_TYPE_VARIANTS
    print(f"\nBaseline measurement ({args.bench_timeout}s total, "
          f"{per_state:.1f}s per type)...")
    baseline_value, baseline_per_variant, unit, _ = run_bench(
        args.bench_timeout, args.metric, aggregate
    )
    if baseline_value is None:
        sys.exit("Baseline benchmark failed.\n")

    baseline_label = "continued" if args.continue_run else "baseline"
    print(f"\n{baseline_label.capitalize()}: {baseline_value:.4f} {unit}")
    for vname, vval in baseline_per_variant:
        print(f"  {vname}: {vval:.4f} {unit}")
    log_result(baseline_value, unit, "baseline",
               "continued from previous run" if args.continue_run else "initial kernel")

    best_value       = baseline_value
    best_per_variant = baseline_per_variant
    best_kernel      = KERNEL_FILE.read_text()

    # --- optimization loop ---
    limit = args.iterations
    failed_kernel: str | None = None
    failure_reason = ""
    failure_output = ""

    for i in count(iter_offset + 1):
        if i - iter_offset > limit:
            break
        sep = "=" * 60
        print(f"\n{sep}")
        tag = f"{i}" if math.isinf(limit) else f"{i}/{int(limit)}"
        print(f"Iteration {tag} best so far: {best_value:.4f} {unit}")
        print(sep)

        response = ask_llm(
            client, args.model, i, args.metric, unit, aggregate,
            failed_kernel=failed_kernel,
            failure_reason=failure_reason,
            failure_output=failure_output,
            direction=args.direction,
        )
        failed_kernel = None
        failure_reason = ""
        failure_output = ""

        new_kernel  = extract_kernel(response)
        description = extract_description(response)

        if new_kernel is None:
            print("No <kernel> block found in LLM response - skipping.")
            log_result(None, unit, "parse_error", "no <kernel> block in response")
            continue

        print(f"\nTrying: {description}\n")
        KERNEL_FILE.write_text(new_kernel)

        build_ok, build_err = cmake_build()
        if not build_ok:
            log_result(None, unit, "build_error", description)
            failed_kernel = new_kernel
            failure_reason = "build_error"
            failure_output = build_err
            KERNEL_FILE.write_text(best_kernel)
            print("Build failed - reverted to best known kernel.")
            continue

        value, per_variant, _, bench_err = run_bench(
            args.bench_timeout, args.metric, aggregate
        )
        if value is None:
            log_result(None, unit, "runtime_error", description)
            failed_kernel = new_kernel
            failure_reason = "runtime_error"
            failure_output = bench_err
            KERNEL_FILE.write_text(best_kernel)
            print("Benchmark/validation failed - reverted.")
            continue

        delta, delta_pct = metric_delta_str(args.metric, value, best_value)

        accept, reasoning = ask_llm_decision(
            client, args.model, args.metric, unit, aggregate,
            best_kernel, new_kernel, description,
            best_per_variant, per_variant,
            best_value, value,
        )
        print(f"\n  LLM decision: {'ACCEPT' if accept else 'REJECT'}")
        print(f"  Reasoning: {reasoning}")

        if accept:
            status           = "accepted"
            best_value       = value
            best_per_variant = per_variant
            best_kernel      = new_kernel
            git_commit_kernel(description, i)
            sign = "faster" if args.metric == "time" else "higher"
            print(f"  {value:.4f} {unit}  "
                  f"({delta} {unit}, {delta_pct:+.1f}% {sign})")
        else:
            status = "rejected"
            KERNEL_FILE.write_text(best_kernel)
            print(f"  {value:.4f} {unit}  "
                  f"({delta} {unit}, {delta_pct:+.1f}%) - reverted")

        log_result(value, unit, status, description)

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Done.  Best: {best_value:.4f} {unit}")
    print(f"Full results: {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
