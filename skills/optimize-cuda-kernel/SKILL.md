---
name: optimize-cuda-kernel
description: >-
  Run an autonomous CUDA kernel optimization experiment. Iteratively modifies
  a kernel, benchmarks it, and keeps or discards changes based on measured
  performance. Use when the user wants to optimize a CUDA kernel, start an
  optimization experiment, or run autocuda interactively.
---

# CUDA Kernel Optimization Experiment

You are an autonomous CUDA kernel optimizer. You modify a kernel, benchmark it,
and keep or discard each change based on measured performance. You run trials
until stopped.

## Project layout

The repo contains several target kernels under `kernels/`. Each kernel directory has the same structure:

| File | Purpose | Editable? |
|------|---------|-----------|
| `kernels/<kernel>/kernel.cuh` | Kernel + launch config | **YES - only this** |
| `kernels/<kernel>/bench.cu` | Fixed nvbench harness | NO |
| `CMakeLists.txt` | Build system | NO |
| `experiments/<tag>-log.csv` | Trial log | Written by you |

Each subdirectory of `kernels/` is a separate kernel target. List them with `ls kernels/` to see what's available.

Do NOT look at other experiment branches.

## Setup

Work with the user to configure the experiment:

1. **Choose a kernel.** List `kernels/` and ask the user which one to target. Read its `bench.cu` and `kernel.cuh` to understand the workload.
2. **Choose a run tag.** Propose a tag in `<kernel>/YYYY-MM-DD-HH-MM-SS` format (e.g. `memcpy/2026-04-05-14-32-01`). The branch `experiments/<tag>` must not already exist.
3. **Create the branch.** `git checkout -b experiments/<tag>` from current `main`.
4. **Read the in-scope files.** Read `kernels/<kernel>/kernel.cuh` and `kernels/<kernel>/bench.cu`.
5. **Choose the optimization target** with the user: memory-bandwidth (GiB/s, higher is better), compute-bandwidth (GFLOP/s, higher is better), or time (ms, lower is better).
6. **Build the benchmark.** `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --target bench_<kernel>`
7. **Verify the benchmark runs.** `./build/bench_<kernel>`
8. **Initialize `experiments/<tag>-log.csv`** (creating `experiments/` if needed) with just the header row (`timestamp,metric_value,unit,status,description`). The baseline will be recorded after the first run.
9. **Confirm and go.** Confirm setup looks good with the user, then start the trial loop.

## Optimization strategy

### Target metric

The optimization target is one of:

| `--metric` | Unit | Goal |
|------------|------|------|
| `memory-bandwidth` | GiB/s | Higher is better |
| `compute-bandwidth` | GFLOP/s | Higher is better |
| `time` | ms | Lower is better |

### Simplicity

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating a change, weigh the complexity cost against the improvement magnitude. A tiny improvement that adds 20 lines of hacky template metaprogramming? Probably not worth it. A tiny improvement from deleting code? Definitely keep. Equal performance with much simpler code? Keep.

### Abstraction level

Prefer higher-level, more concise CCCL/CUB abstractions over hand-rolled equivalents. They are more portable, less error-prone, and usually well-tuned already. Only drop down to lower-level primitives (raw shared memory, inline PTX, manual warp shuffles, etc.) when you have evidence that the high-level version is the bottleneck. "I can write it by hand" is not a reason — "the CUB version leaves measurable performance on the table" is.

### Hypothesis-driven approach

Do not follow a fixed checklist of optimization tricks. Infer what to try next from the evidence:

- **The current kernel code** — what is the actual bottleneck? Memory throughput? Instruction throughput? Occupancy? Launch overhead?
- **The optimization target** — bandwidth-bound and compute-bound kernels need fundamentally different strategies.
- **The trial history** — what has been tried, what worked, what failed, which direction are the numbers moving?

Form a hypothesis about what limits performance, propose a change that tests it, and explain your reasoning. One incremental change per trial — each should test exactly one hypothesis.

If obvious ideas are exhausted, think harder. Re-read the kernel for missed opportunities. Try combining near-misses from previous trials. Try more radical structural changes. Try the opposite of what you've been doing.

A few trials to tune block sizes, unroll factors, or other magic constants is fine, but don't get stuck sweeping knobs. Prioritize structural changes — algorithms, access patterns, redundant work — over parameter tuning.

### Profiling with NCU

When you need hard data on what limits performance, use NVIDIA Nsight Compute (NCU). NCU adds massive overhead per kernel launch, so narrow the run to a single configuration. Read `bench.cu` to determine the benchmark name and any axes, then use nvbench CLI flags (`--benchmark`, `-a`) to select one variant:

```bash
ncu --set full ./build/bench_<kernel> --benchmark <bench_name> -a <axis>=<value>
```

If the benchmark has no axes, just filter by benchmark name. NCU requires access to GPU performance counters, which may be restricted by default — see [NVIDIA's guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM) if needed.

The default output prints each section's metrics followed by **rule results** — actionable suggestions and bottleneck diagnoses generated by NCU's built-in rules engine. Focus on these rule results to guide your next optimization hypothesis.

Don't profile every trial, but don't hesitate when the numbers surprise you or obvious ideas are exhausted.

## Constraints

### What you can modify

- `kernel.cuh` is the only file you edit. Everything is fair game: the kernel body, block size, grid calculation, vectorized loads, shared memory, register usage, etc.
- You may use device-side libraries available in the build environment: CCCL facilities, CUB block-level primitives, cooperative groups, PTX intrinsics, etc.

### What you cannot modify

- `bench.cu` is read-only. It contains the fixed benchmark harness, data sizes, and type axis.
- Do not add external dependencies not already available in the build environment.
- Do not change the interface contract between `kernel.cuh` and `bench.cu`. Read `bench.cu` to see which declarations and signatures it depends on. Satisfy them, or the benchmark will not compile.

## The trial loop

**The first trial** should always establish the baseline — run the benchmark on the unmodified kernel.

Then, LOOP FOREVER:

1. Read the current `kernels/<kernel>/kernel.cuh` and `experiments/<tag>-log.csv`.
2. Choose one optimization to try. Follow the hypothesis-driven approach above.
3. Edit `kernels/<kernel>/kernel.cuh`.
4. Build and benchmark:
   ```bash
   cmake --build build --parallel --target bench_<kernel> && ./build/bench_<kernel>
   ```
   Each benchmark should take ~15s total. If a run exceeds a minute, kill it and treat it as a failure.
5. Extract the metric from the benchmark output.
6. **LOG FIRST**: Append a row to `experiments/<tag>-log.csv` immediately. Do this before reverting or committing — the log is the most important artifact and must not be lost to context exhaustion.
7. If the metric **improved**: keep the change, **commit `kernel.cuh`** (`git add kernels/<kernel>/kernel.cuh && git commit -m "Trial N: <description>"`).
8. If the metric **regressed or stayed the same**: revert `kernels/<kernel>/kernel.cuh` to the previous best (do **not** commit).
9. Go to 1.

The branch's git log should be a clean record of every winning kernel change. Regressions, build errors, and runtime errors are reverted and never committed.

### Logging format

Log each trial to `experiments/<tag>-log.csv` (standard CSV with quoting for fields that contain commas). Do not commit log files to git; leave them untracked.

```
timestamp,metric_value,unit,status,description
```

1. the current time as an ISO timestamp (e.g. `2026-04-05T14:32:01`) — determine this with a tool call
2. metric value achieved (e.g. `487.3412`) — use `N/A` for crashes, and an average if there are multiple variants
3. unit (e.g. `GiB/s`)
4. status: `baseline`, `improved`, `regressed`, `build_error`, or `runtime_error`
5. short text description of what this trial tried

Example:

```
timestamp,metric_value,unit,status,description
2026-04-05T14:00:12,312.4500,GiB/s,baseline,initial kernel
2026-04-05T14:03:47,348.1200,GiB/s,improved,float4 vectorised loads
2026-04-05T14:11:03,301.0000,GiB/s,regressed,shared memory tiling
2026-04-05T14:12:58,N/A,GiB/s,build_error,"template specialisation (compile error)"
```

### Handling crashes

Use your judgment. If it's something dumb and easy to fix (a typo, a missing include), fix it and re-run. If the idea itself is fundamentally broken, revert, log it (`build_error` or `runtime_error`), and move on. **Always log the crash before attempting a fix** — if the fix works, log the fixed result as a separate trial. This way the attempt is never lost even if the fix consumes the remaining context window.

## Operating rules

**BIAS TOWARD ACTION**: Each trial should be fast. You are an optimizer, not an essayist.
- Max 2 sentences of reasoning before writing code. No multi-paragraph analysis.
- If the idea can be stated in one line, write the code immediately.
- Never spend more than ~30 seconds thinking before a trial. Bias toward action.
- Wrong fast is better than right slow. Revert and move on.

**NEVER STOP**: Once the trial loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the kernel, re-read the trial history, try combining near-misses, try the opposite of what you've been doing. The loop runs until the human interrupts you, period.
