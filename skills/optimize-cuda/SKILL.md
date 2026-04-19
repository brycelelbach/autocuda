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
- What metric to optimize, its unit, and direction.
- Timeout limits for builds and benchmarks.

## Setup

1. **Read `project-layout.md`.** Understand the project structure, editable files, build commands, validation commands, benchmark commands, metric, and timeouts.
2. **Choose a run tag.** Unless otherwise specified, use the starting date and time in the `YYYY-MM-DD-HH-MM-SS` format (e.g. `2026-04-05-14-32-01`) for the tag name. The branch `experiment/<tag>` must not already exist.
3. **Create the branch.** `git checkout -b experiment/<tag>` from the current branch.
4. **Read the editable source files** listed in `project-layout.md`.
5. **Build, validate, and benchmark.** Run the commands from `project-layout.md` to verify everything works. This establishes the baseline.
6. **Initialize the trial log** at `experiments/<tag>-log.csv` (creating `experiments/` if needed) with the header row (`timestamp,metric_value,unit,status,description`) and a baseline entry from step 5.
7. **Start the trial loop.**

## Optimization strategy

### Simplicity

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating a change, weigh the complexity cost against the improvement magnitude. A tiny improvement that adds 20 lines of hacky template metaprogramming? Probably not worth it. A tiny improvement from deleting code? Definitely keep. Equal performance with much simpler code? Keep.

### Abstraction level

Prefer higher-level libraries and abstractions over hand-rolled equivalents when they are available in the build environment. They are more portable, less error-prone, and usually well-tuned. Only drop down to lower-level primitives when you have evidence that the high-level version is the bottleneck. "I can write it by hand" is not a reason — "the library version leaves measurable performance on the table" is.

The CUDA ecosystem offers libraries at many levels. Prefer the highest level that meets your performance needs:

- **CCCL (Thrust / CUB / libcudacxx)** — parallel algorithms, block/warp-level primitives, C++ standard library facilities for device code.
- **CUTLASS** — templated GEMM and convolution building blocks; prefer over hand-rolled matrix math.
- **cuBLAS / cuBLASLt** — dense linear algebra; hard to beat for standard BLAS operations.
- **cuDNN** — deep learning primitives (convolutions, normalization, attention); use when the workload fits.
- **cuFFT / cuSPARSE / cuRAND / cuSOLVER** — domain-specific libraries for FFTs, sparse linear algebra, random number generation, and dense solvers.
- **Cooperative groups** — flexible thread grouping beyond the traditional block/warp model.
- **PTX intrinsics / inline assembly** — last resort for squeezing out final percentage points when profiling proves it necessary.

### Hypothesis-driven approach

Do not follow a fixed checklist of optimization tricks. Infer what to try next from the evidence:

- **The current source code** — what is the actual bottleneck? Memory throughput? Instruction throughput? Occupancy? Launch overhead? Host-device transfer? Synchronization?
- **The metric** — bandwidth-bound, compute-bound, and latency-bound workloads need fundamentally different strategies.
- **The trial history** — what has been tried, what worked, what failed, which direction are the numbers moving?

Form a hypothesis about what limits performance, propose a change that tests it, and explain your reasoning. One incremental change per trial — each should test exactly one hypothesis.

If obvious ideas are exhausted, think harder. Re-read the source for missed opportunities. Try combining near-misses from previous trials. Try more radical structural changes. Try the opposite of what you've been doing.

A few trials to tune block sizes, unroll factors, or other magic constants is fine, but don't get stuck sweeping knobs. Prioritize structural changes — algorithms, access patterns, redundant work — over parameter tuning.

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

## The trial loop

**The first trial** should always establish the baseline — validate and benchmark the unmodified code.

Then, LOOP FOREVER:

1. Re-read the editable source files and `experiments/<tag>-log.csv`.
2. Choose one optimization to try. Follow the hypothesis-driven approach above.
3. Edit the source file(s).
4. Build. If the build fails, see **Handling failures** below.
5. Validate. If validation fails, see **Handling failures** below.
6. Benchmark. If the benchmark times out or crashes, see **Handling failures** below. Use the timeouts from `project-layout.md`.
7. Extract the metric from the benchmark output as described in `project-layout.md`.
8. **LOG FIRST**: Append a row to `experiments/<tag>-log.csv` immediately. Do this before reverting or committing — the log is the most important artifact and must not be lost to context exhaustion.
9. If the metric **improved**: keep the change, **commit the modified files** (`git add <files> && git commit -m "Trial N: <description>"`).
10. If the metric **regressed or stayed the same**: revert all modified files to the previous best (do **not** commit).
11. Go to 1.

The branch's git log should be a clean record of every winning change. Regressions, build errors, validation failures, and runtime errors are reverted and never committed.

### Logging format

Log each trial to `experiments/<tag>-log.csv` (standard CSV with quoting for fields that contain commas). Do not commit log files to git; leave them untracked.

```
timestamp,metric_value,unit,status,description
```

1. the current time as an ISO timestamp (e.g. `2026-04-05T14:32:01`) — determine this with a tool call
2. metric value achieved (e.g. `487.3412`) — use `N/A` for failures
3. unit (e.g. `GiB/s`, `ms`, `GFLOP/s`)
4. status: `baseline`, `improved`, `regressed`, `build_error`, `validation_error`, or `runtime_error`
5. short text description of what this trial tried

Example:

```
timestamp,metric_value,unit,status,description
2026-04-05T14:00:12,312.4500,GiB/s,baseline,initial code
2026-04-05T14:03:47,348.1200,GiB/s,improved,float4 vectorised loads
2026-04-05T14:11:03,301.0000,GiB/s,regressed,shared memory tiling
2026-04-05T14:15:22,N/A,GiB/s,validation_error,loop unrolling broke edge case
2026-04-05T14:12:58,N/A,GiB/s,build_error,"template specialisation (compile error)"
```

### Handling failures

Use your judgment. If it's something dumb and easy to fix (a typo, a missing include, an off-by-one in a boundary check), fix it and re-run. If the idea itself is fundamentally broken, revert, log it, and move on.

**Always log the failure before attempting a fix** — if the fix works, log the fixed result as a separate trial. This way the attempt is never lost even if the fix consumes the remaining context window.

Failure types:

- **`build_error`** — compilation or linking failure. Revert, log, move on (or fix if trivial).
- **`validation_error`** — the code compiles and runs but produces incorrect results. This is serious — revert immediately. Never keep a change that breaks correctness.
- **`runtime_error`** — crash, hang, or timeout during benchmarking. Revert, log, move on.

## Operating rules

**BIAS TOWARD ACTION**: Each trial should be fast. You are an optimizer, not an essayist.
- Max 2 sentences of reasoning before writing code. No multi-paragraph analysis.
- If the idea can be stated in one line, write the code immediately.
- Never spend more than ~30 seconds thinking before a trial. Bias toward action.
- Wrong fast is better than right slow. Revert and move on.

**NEVER STOP**: Once the trial loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the source, re-read the trial history, try combining near-misses, try the opposite of what you've been doing. The loop runs until the human interrupts you, period.
