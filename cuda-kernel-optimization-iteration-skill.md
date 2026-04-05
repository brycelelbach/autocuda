# CUDA Kernel Optimization Iteration

You are an autonomous CUDA kernel optimizer. You iteratively modify a kernel,
benchmark it, and keep or discard each change based on measured performance. You
run until stopped.

## Project layout

| File | Purpose | Editable? |
|------|---------|-----------|
| `kernel.cuh` | Kernel + launch config | **YES - only this** |
| `bench.cu` | Fixed nvbench harness | NO |
| `CMakeLists.txt` | Build system | NO |
| `results.csv` | Experiment log | Written by you |
| `cuda-kernel-optimization-idea-skill.md` | Goals, constraints, and optimization philosophy. Do this every iteration. | NO |

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag:** propose a tag based on today's date (e.g. `2026-04-05`). The branch `autoresearch/<tag>` must not already exist - this is a fresh run.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from current `main`. Every successful experiment will be committed to this branch (see **The experiment loop** below).
3. **Read the in-scope files.** The repo is small; read the files listed in project layout.
4. **Agree on the optimization target** with the user: bandwidth (GiB/s, higher is better), time (ms, lower is better), or FLOP/s (GFLOP/s, higher is better).
5. **Build the benchmark:** `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
6. **Verify the benchmark runs:** `./build/bench`
7. **Initialize `results.csv`** with just the header row (`timestamp,metric_value,unit,status,description`). The baseline will be recorded after the first run.
8. **Confirm and go.** Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is a single edit-build-benchmark cycle. The benchmark runs for a fixed time budget (~15s total across all type variants). You build and run simply as:

```bash
cmake --build build --parallel && ./build/bench
```

The goals, constraints, simplicity criterion, abstraction-level preference, and hypothesis-driven approach are all defined in `cuda-kernel-optimization-idea-skill.md`. Read it and follow it.

**The first run**: Your very first run should always be to establish the baseline - run the benchmark on the unmodified kernel.

## Logging results

When an experiment is done, log it to `results.csv` (standard CSV with quoting for fields that contain commas). Do not commit `results.csv` to git; leave it untracked.

The CSV has a header row and 5 columns:

```
timestamp,metric_value,unit,status,description
```

1. ISO timestamp (e.g. `2026-04-05T14:32:01`)
2. metric value achieved (e.g. `487.3412`) - use `N/A` for crashes
3. unit (e.g. `GiB/s`)
4. status: `baseline`, `improved`, `regressed`, `build_error`, or `runtime_error`
5. short text description of what this experiment tried

Example:

```
timestamp,metric_value,unit,status,description
2026-04-05T14:00:00,312.4500,GiB/s,baseline,initial kernel
2026-04-05T14:05:00,348.1200,GiB/s,improved,float4 vectorised loads
2026-04-05T14:10:00,301.0000,GiB/s,regressed,shared memory tiling
2026-04-05T14:15:00,N/A,GiB/s,build_error,"template specialisation (compile error)"
```

## The experiment loop

LOOP FOREVER:

1. Read the current `kernel.cuh` and `results.csv`.
2. Choose one optimization to try. Follow the hypothesis-driven approach from `cuda-kernel-optimization-idea-skill.md`.
3. Edit `kernel.cuh`.
4. Build and benchmark:
   ```bash
   cmake --build build --parallel && ./build/bench
   ```
5. Extract the metric from the benchmark output.
6. If the benchmark crashed, see **Crashes** below.
7. If the metric **improved**: keep the change, **commit `kernel.cuh`** (`git add kernel.cuh && git commit -m "iteration N: <description>"`), and record as `improved`.
8. If the metric **regressed or stayed the same**: revert `kernel.cuh` to the previous best (do **not** commit). Record as `regressed`.
9. Append a row to `results.csv`.
10. Go to 1.

The branch's git log should be a clean record of every winning kernel change. Regressions, build errors, and runtime errors are reverted and never committed.

**Timeout**: Each benchmark should take ~15s total. If a run exceeds a minute, kill it and treat it as a failure (revert and log as `runtime_error`).

**Crashes**: Use your judgment. If it's something dumb and easy to fix (a typo, a missing include), fix it and re-run. If the idea itself is fundamentally broken, revert, log it (`build_error` or `runtime_error`), and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder - re-read the kernel, re-read the experiment history, try combining near-misses, try the opposite of what you've been doing. The loop runs until the human interrupts you, period.
