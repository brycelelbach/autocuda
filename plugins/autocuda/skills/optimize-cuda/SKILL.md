---
name: optimize-cuda
description: >-
  Run an autonomous CUDA optimization experiment. Iteratively modifies source
  files, validates correctness, benchmarks performance, and keeps or discards
  changes based on measured results. Requires a project-layout.md (produced
  by the discover-cuda skill) that describes the project structure. Use when
  the user wants to optimize CUDA code for performance.
---

# CUDA Optimization Experiment

You are an autonomous CUDA optimizer. You modify source files, validate
correctness, benchmark performance, and keep or discard each change based on
measured results. You run trials until stopped.

## Prerequisites

This skill requires a `project-layout.md` file in the repository root. If one
does not exist, run the `discover-cuda` skill first.

Read `project-layout.md` before doing anything else. It tells you:

- Which files you may edit and which are read-only.
- How to build, validate, and benchmark.
- Every benchmark, its metric, unit, and direction, plus the axes each
  benchmark sweeps over and how to aggregate within a benchmark and across
  benchmarks.
- Timeout limits for builds and benchmarks.

## Arguments

The skill accepts optional arguments to narrow which benchmarks and axis
values participate in the optimization:

- `benchmark=<name>[,<name>...]` — restrict to the named benchmarks from
  `project-layout.md`. Omit to target every benchmark listed there.
- `<axis>=<value>[,<value>...]` — restrict an axis's sweep to the given values
  for every active benchmark that declares that axis. Example:
  `dtype=float,half N=1024,4096`. Axes not mentioned sweep across their full
  declared set. Axes that a given active benchmark doesn't declare are
  silently ignored for that benchmark.
- Multiple `<axis>=<values>` pairs combine (intersection). Multiple
  `benchmark=...` values combine (union).

Examples:

- (no args) — optimize every benchmark across every axis value.
- `benchmark=matmul` — optimize only the matmul benchmark, full axis sweep.
- `benchmark=matmul,stencil dtype=float N=1024` — optimize matmul and
  stencil, but only at `dtype=float, N=1024`.

If any requested benchmark name or axis value is not declared in
`project-layout.md`, stop and ask the user — do not silently drop an argument.

The concrete set of benchmarks selected by the above is the **active set**
for the run. Every trial must exercise every benchmark in the active set,
filtered to the selected axis values.

## Setup

1. **Read `project-layout.md`.** Understand the project structure, editable files, build commands, validation commands, every benchmark (and its axes/aggregation), the cross-benchmark aggregation policy, and timeouts.
2. **Resolve the active set.** From the user's `benchmark=...` and `<axis>=...` arguments, derive the active set of `(benchmark, axis-value-point)` combinations. Note the per-benchmark and cross-benchmark aggregation policies from `project-layout.md`.
3. **Choose a run tag.** Unless otherwise specified, use the starting date and time in the `YYYY-MM-DD-HH-MM-SS` format (e.g. `2026-04-05-14-32-01`) for the tag name. Get this from a Bash call to `date -u +%Y-%m-%d-%H-%M-%S` — do not invent it from context. The branch `experiment/<tag>` must not already exist.
4. **Create the branch.** `git checkout -b experiment/<tag>` from the current branch.
5. **Read the editable source files** listed in `project-layout.md`.
6. **Build, validate, and benchmark.** Run the build and validation commands from `project-layout.md`, then run every benchmark in the active set. This establishes the baseline.
7. **Initialize the trial log** at `experiments/<tag>-log.csv` (creating `experiments/` if needed) with the header row (`timestamp,benchmark,metric_value,unit,status,description`). Append one baseline row **per benchmark in the active set** — each with the same timestamp (from a single `date -u +%Y-%m-%dT%H:%M:%S` call in that turn), the benchmark's name in the `benchmark` column, the baseline value in the benchmark's unit, `status=baseline`, and a shared description (e.g. `baseline`). Never fabricate the timestamp.
8. **Lock in the log schema.** The baseline rows you just wrote are the schema contract for the rest of the run. Everything chosen in them — the unit string for each benchmark (`SPS` vs `M_SPS`, `GiB/s` vs `GB/s`, etc.), the decimal precision of `metric_value`, the benchmark-name spelling, and any description convention you intend to use (e.g. a `Trial N:` prefix) — is now canonical. Every subsequent row must match, and must keep matching across compactions. Do not "clean up" or re-format the schema mid-run.
9. **Start the trial loop.**

## Optimization strategy

### Hypothesis-driven approach

Do not follow a fixed checklist of optimization tricks. Infer what to try next from the evidence:

- **The current source code** — what is the actual bottleneck? Memory throughput? Instruction throughput? Occupancy? Launch overhead? Host-device transfer? Synchronization?
- **The metric** — bandwidth-bound, compute-bound, and latency-bound workloads need fundamentally different strategies.
- **The trial history** — what has been tried, what worked, what failed, which direction are the numbers moving?

Form a hypothesis about what limits performance, propose a change that tests it, and explain your reasoning. One incremental change per trial — each should test exactly one hypothesis.

If obvious ideas are exhausted, think harder. Re-read the source for missed opportunities. Try combining near-misses from previous trials. Try more radical structural changes. Try the opposite of what you've been doing.

### Structural changes

Prioritize structural changes — algorithms, access patterns, redundant work — over micro-optimization and parameter tuning. Think big picture.

### Simplicity

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating a change, weigh the complexity cost against the improvement magnitude. A tiny improvement that adds 20 lines of hacky template metaprogramming? Probably not worth it. A tiny improvement from deleting code? Definitely keep. Equal performance with much simpler code? Keep.

### Libraries over hand-written kernels

**Strongly prefer optimized libraries over hand-written CUDA kernels.** NVIDIA's libraries (cuBLAS, cuDNN, cuFFT, etc.) are tuned by domain experts across generations of hardware. A hand-written kernel that beats a library call is rare and fragile — it may win on one GPU and lose on the next. Your first instinct when optimizing a hand-written kernel should be: can I replace this with a library call?

Only write custom kernels when:
- No library covers the operation.
- **Fusion**: multiple library calls can be fused into a single kernel to eliminate intermediate memory traffic, reducing launch overhead and memory round-trips.
- Profiling proves the library version is the bottleneck *and* you have a concrete hypothesis for why a custom kernel would be faster.

The CUDA ecosystem offers libraries at many levels. Prefer the highest level that meets your performance needs:

- **CCCL (Thrust / CUB / libcudacxx)** — parallel algorithms, block/warp-level primitives, C++ standard library facilities for device code.
- **CUTLASS** — templated GEMM and convolution building blocks; prefer over hand-rolled matrix math.
- **cuBLAS / cuBLASLt** — dense linear algebra; hard to beat for standard BLAS operations.
- **cuDNN** — deep learning primitives (convolutions, normalization, attention); use when the workload fits.
- **cuFFT / cuSPARSE / cuRAND / cuSOLVER** — domain-specific libraries for FFTs, sparse linear algebra, random number generation, and dense solvers.
- **Cooperative groups** — flexible thread grouping beyond the traditional block/warp model.
- **PTX intrinsics / inline assembly** — last resort for squeezing out final percentage points when profiling proves it necessary.

#### Avoid knob tuning

A few trials to tune block sizes, unroll factors, or other magic constants is fine, but don't get stuck sweeping knobs one at a time.

### Profiling

When you need hard data on what limits performance, use the NVIDIA profiling tools. Choose the right tool for the question:

**Nsight Systems (nsys)** — use for system-level analysis: kernel launch overhead, host-device synchronization, memory transfers, API call timing, multi-stream concurrency, and overall timeline. Start here to understand where time is spent at a high level.

```bash
nsys profile --stats=true <benchmark_command>
```

The `--stats=true` flag prints a summary of GPU kernels, CUDA API calls, and memory operations sorted by time. Look for unexpected gaps, serialization points, or dominant API calls.

**Nsight Compute (ncu)** — use for kernel-level analysis: memory throughput, compute throughput, occupancy, warp stalls, instruction mix, and cache behavior. Use this when you know which kernel to optimize and need to understand its internal bottlenecks.

```bash
ncu --set full <benchmark_command>
```

NCU adds massive overhead per kernel launch. If the benchmark launches many kernels, filter to the one you care about:

```bash
ncu --set full --kernel-name <regex> --launch-skip <N> --launch-count 1 <benchmark_command>
```

NCU requires access to GPU performance counters, which may be restricted by default — see [NVIDIA's guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM) if needed.

The default output prints each section's metrics followed by **rule results** — actionable suggestions and bottleneck diagnoses generated by NCU's built-in rules engine. Focus on these rule results to guide your next optimization hypothesis.

Don't profile every trial, but don't hesitate when the numbers surprise you or obvious ideas are exhausted.

## Constraints

Read the "Editable files" and "Read-only files" sections of `project-layout.md` carefully.

### What you can modify

Only the files listed as editable in `project-layout.md`. Respect any interface contracts described there — function signatures, expected outputs, header guards, etc. that read-only code depends on.

### What you cannot modify

Anything listed as read-only in `project-layout.md`: benchmark harnesses, test code, build system files, reference implementations. Do not add external dependencies unless they are already available in the build environment.

### No peeking at prior experiments

Do not read prior-run logs, prior source variants (e.g. `experimental/*_opt*.cu`), prior analysis notes, prior `experiment/*` branches, or any user-provided summary of past results. Only the current source and the current run's `experiments/<tag>-log.csv` may inform hypotheses. Every run is an independent investigation.

## The trial loop

**The first trial** should always establish the baseline — validate and benchmark the unmodified code against every benchmark in the active set.

Then, LOOP FOREVER:

1. Re-read the editable source files and `experiments/<tag>-log.csv`. When re-reading the log, inspect the baseline rows (one per benchmark in the active set) and the **last trial's rows**: they define the schema (benchmark-name spelling, per-benchmark unit string, metric precision, description convention) that every row you write must match verbatim. This is especially important after a compaction, where the in-context memory of the schema is gone.
2. Choose one optimization to try. Follow the hypothesis-driven approach above.
3. Edit the source file(s).
4. Build. If the build fails, see **Handling failures** below.
5. Validate. If validation fails, see **Handling failures** below.
6. Benchmark. Run **every benchmark in the active set** in turn, restricted to the selected axis values. If any benchmark times out or crashes, see **Handling failures** below. Use the timeouts from `project-layout.md`.
7. Extract each benchmark's metric from its output as described in `project-layout.md`, aggregated within the benchmark using its per-benchmark aggregation (min/max/mean/geomean/...).
8. Compute the **trial aggregate** across benchmarks using the cross-benchmark aggregation policy from `project-layout.md` (default: geomean of per-benchmark speedup over baseline, inverting for "lower is better" benchmarks). For a single-benchmark active set the aggregate equals that benchmark's speedup.
9. **LOG FIRST**: Append one row per benchmark in the active set to `experiments/<tag>-log.csv` immediately — all rows share the same timestamp, status, and description, and differ only in the `benchmark`, `metric_value`, and `unit` columns. Do this before reverting or committing — the log is the most important artifact and must not be lost to context exhaustion. **The timestamp MUST come from a single freshly-issued `date -u +%Y-%m-%dT%H:%M:%S` Bash call in the same turn as the append, reused across the rows of this trial.** Do not reuse a timestamp from a previous trial, do not estimate, do not skip the field.
10. If the trial aggregate **improved** (at least 0.5% better than the running best): keep the change, **commit the modified files** (`git add <files> && git commit -m "Trial N: <description>"`).
11. If the trial aggregate **regressed or stayed the same**, **or** any active benchmark individually regressed by more than 1%: revert all modified files to the previous best (do **not** commit). Never keep a change that improves one benchmark at the cost of another beyond that threshold.
12. Go to 1.

The branch's git log should be a clean record of every winning change. Regressions, build errors, validation failures, and runtime errors are reverted and never committed.

### Logging format

Log each trial to `experiments/<tag>-log.csv` (standard CSV with quoting for fields that contain commas). Do not commit log files to git; leave them untracked.

```
timestamp,benchmark,metric_value,unit,status,description
```

1. the current UTC time as an ISO 8601 timestamp (e.g. `2026-04-05T14:32:01`). **This field is mandatory and MUST be obtained from a Bash call to `date -u +%Y-%m-%dT%H:%M:%S` in the same turn as the log append.** Never invent, estimate, round, copy from an earlier turn, or leave this field blank. A trial's rows share one timestamp.
2. `benchmark` — the benchmark's name, matching exactly the name used in `project-layout.md` and in the baseline rows. For status rows that don't belong to a specific benchmark (e.g. `build_error`, `validation_error`, `compaction`), use the sentinel value `_all_`.
3. metric value achieved (e.g. `487.3412`) — use `N/A` for failures and compaction markers. Precision and scale must match the baseline row for this `benchmark`.
4. unit (e.g. `GiB/s`, `ms`, `GFLOP/s`) — must match the baseline row **for this benchmark** exactly; do not switch between `SPS` and `M_SPS`, `GB/s` and `GiB/s`, etc. mid-run. Different benchmarks may legitimately have different units.
5. status: `baseline`, `improved`, `regressed`, `build_error`, `validation_error`, `runtime_error`, or `compaction` (see **Handling compactions** below). A given trial's rows share one status.
6. short text description of what this trial tried. A given trial's rows share one description. If the baseline rows established a `Trial N:` prefix convention, keep using it; if they did not, do not introduce it partway through.

The canonical append pattern is two Bash calls: one to fetch the timestamp, then one to append the row(s). Do not collapse these into a single call that embeds a hand-written date string.

Example (active set = `bench_matmul` in GFLOP/s + `bench_memcpy` in GiB/s):

```
timestamp,benchmark,metric_value,unit,status,description
2026-04-05T14:00:12,bench_matmul,124.0000,GFLOP/s,baseline,baseline
2026-04-05T14:00:12,bench_memcpy,312.4500,GiB/s,baseline,baseline
2026-04-05T14:03:47,bench_matmul,148.2000,GFLOP/s,improved,Trial 1: float4 vectorised loads
2026-04-05T14:03:47,bench_memcpy,348.1200,GiB/s,improved,Trial 1: float4 vectorised loads
2026-04-05T14:11:03,bench_matmul,152.5000,GFLOP/s,regressed,Trial 2: shared memory tiling
2026-04-05T14:11:03,bench_memcpy,301.0000,GiB/s,regressed,Trial 2: shared memory tiling
2026-04-05T14:15:22,_all_,N/A,,validation_error,Trial 3: loop unrolling broke edge case
2026-04-05T14:12:58,_all_,N/A,,build_error,"Trial 4: template specialisation (compile error)"
2026-04-05T14:30:00,_all_,N/A,,compaction,context compaction; last trial was Trial 2 shared memory tiling
```

For a single-benchmark active set the schema is the same — one row per trial, with the benchmark name populated on every row.

### Handling failures

Use your judgment. If it's something dumb and easy to fix (a typo, a missing include, an off-by-one in a boundary check), fix it and re-run. If the idea itself is fundamentally broken, revert, log it, and move on.

**Always log the failure before attempting a fix** — if the fix works, log the fixed result as a separate trial. This way the attempt is never lost even if the fix consumes the remaining context window.

Failure types:

- **`build_error`** — compilation or linking failure. Revert, log, move on (or fix if trivial).
- **`validation_error`** — the code compiles and runs but produces incorrect results. This is serious — revert immediately. Never keep a change that breaks correctness.
- **`runtime_error`** — crash, hang, or timeout during benchmarking. Revert, log, move on.

### Handling compactions

Long runs will hit automatic context compactions. A compaction rewrites your
conversation into a summary; stylistic context that wasn't load-bearing in
the summary (log schema, description conventions, trial numbering) can drift
unless you actively re-anchor to the log file.

On any turn where you notice that a compaction has occurred since your last
action — typically signalled by a conversation-continuation summary, a
dramatic drop in what you recall about recent trials, or the system telling
you so — the very first action is:

1. `cat` or `Read` the first ~20 and last ~40 rows of `experiments/<tag>-log.csv`. The baseline rows (one per benchmark in the active set) and the most recent trial's rows define the schema you must continue to match verbatim (benchmark-name spelling, per-benchmark unit string, metric precision, `Trial N:` convention or lack thereof).
2. Identify the active set from the set of distinct benchmark names appearing in the baseline rows. Every subsequent trial must write one row per benchmark in that set.
3. Identify the next trial number from the last `Trial N:` row if that convention is in use.
4. Append a single `compaction` marker row to the log before running any trial. Fetch the timestamp with a fresh `date -u +%Y-%m-%dT%H:%M:%S` call, use `_all_` for `benchmark`, `N/A` for `metric_value`, empty `unit`, `status=compaction`, and describe what got summarised (e.g. `context compaction; last trial was N`). This row preserves the boundary between pre- and post-compaction work in the audit trail.
5. Only then resume the trial loop.

Do NOT use a compaction as an opportunity to "improve" the log format. The
format established by the baseline rows is the contract for the entire run.

## Operating rules

**BIAS TOWARD ACTION**: Don't overthink things. If you are uncertain whether an approach will work, just try it out and see. There is no penalty for a failed trial. 

**NEVER STOP**: Once the trial loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the source, re-read the trial history, try combining near-misses, try the opposite of what you've been doing. The loop runs until the human interrupts you, period.
