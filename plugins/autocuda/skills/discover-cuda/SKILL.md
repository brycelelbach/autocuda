---
name: discover-cuda
description: >-
  Discover the structure of a CUDA project: locate source files, validation
  code, benchmarks, and build system. Set up the environment, verify
  everything builds and runs, and write a project-layout.md that the
  optimize-cuda skill consumes. Use when starting work on a new or unfamiliar
  CUDA project.
---

# Discover CUDA Project

You are a CUDA project analyst. You explore an unfamiliar codebase, identify
the key files, set up the build environment, verify that validation and
benchmarks pass, and produce a structured `project-layout.md` that serves as
the ground truth for subsequent optimization work.

## Discovery

Explore the repository to understand its structure. You are looking for:

1. **Source files** — CUDA kernels (`.cu`, `.cuh`), host code, headers. Identify which files contain the performance-critical code that is the target of optimization.
2. **Validation code** — tests, correctness checks, reference implementations. These verify that optimized code still produces correct results. They cannot be modified during optimization.
3. **Benchmark code** — performance measurement harnesses, timing scripts, benchmark drivers. These produce the metric to optimize. They cannot be modified during optimization.
4. **Build system** — CMake, Makefiles, Bazel, shell scripts, etc. Understand how to configure and build the project.
5. **Dependencies** — CUDA toolkit version requirements, third-party libraries (Thrust, CUB, CUTLASS, cuBLAS, cuDNN, cuFFT, cuSPARSE, cuRAND, etc.), Python packages, system libraries.
6. **Metric** — what the benchmark measures (execution time, throughput, bandwidth, FLOP/s, latency, etc.), how it reports results, and what direction is better.

Read READMEs, build scripts, CI configs, and top-level files first. Then drill into directories that look relevant. Ask the user for clarification when the project structure is ambiguous — e.g. when multiple benchmark harnesses exist or it's unclear which files are in scope for optimization.

## Environment setup

Once you understand the project, set up the build environment:

1. **Check prerequisites.** Verify the CUDA toolkit, compiler, and build tools are available. Report the GPU architecture (run `nvidia-smi` and note the compute capability).
2. **Install dependencies.** Use the project's dependency management (CMake's FetchContent, vcpkg, conan, pip, conda, apt, etc.). If the project has a setup script, use it.
3. **Configure the build.** Run the project's configure step (e.g. `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release`). Fix configuration errors — missing packages, wrong paths, etc.
4. **Build.** Compile all targets needed for validation and benchmarking. Note the build commands and any non-obvious flags.
5. **Run validation.** Execute the test suite or correctness checks. Everything must pass before any optimization begins. If tests fail on the unmodified code, stop and report the issue to the user.
6. **Run the benchmark.** Execute the benchmark harness. Record the baseline metric value, the unit, and which direction is better. Note how long the benchmark takes — this establishes the timeout for optimization trials.

## Write project-layout.md

After successful discovery and verification, write a `project-layout.md` file in the repository root. This file is consumed by the `optimize-cuda` skill and must contain all the information needed to run the optimization loop without further discovery.

The file must have the following sections:

### Editable files

List every file (or file pattern) that may be modified during optimization. Be specific — list exact paths, not just directories. Explain what each file contains and any interface contracts it must satisfy (function signatures, expected outputs, etc.).

### Read-only files

List files that must NOT be modified: benchmark harnesses, test code, build system files, reference implementations. Explain what each one does so the optimizer understands the constraints.

### Build

Exact commands to build the project from a clean state and to do an incremental rebuild after modifying source files. Include any necessary environment variables or flags.

### Validation

Exact command(s) to run correctness checks. Describe what a passing run looks like (exit code, expected output pattern, etc.).

### Benchmarks

List **every** benchmark you find — not just the one that looks most important.
A project commonly has more than one (e.g. one benchmark per kernel, or a suite
harness alongside a microbench). `optimize-cuda` needs to know about all of
them so the user can target some or all.

For each benchmark, give its own subsection named after the benchmark itself
(e.g. `#### bench_matmul`), with every one of the following fields:

- **Command** — the exact shell command to run it from the repo root (e.g.
  `./build/bench_matmul`). Include any flags the benchmark needs.
- **Metric** — what it measures (execution time, memory bandwidth, GFLOP/s,
  latency, etc.).
- **Unit** — `ms`, `GiB/s`, `GFLOP/s`, etc.
- **Direction** — higher is better, or lower is better.
- **Metric extraction** — exactly how to pull the metric value from the
  benchmark's output (e.g. "parse the last line", "read the JSON field
  `results.mean_time`", "take the median of the Bandwidth column").
- **Axes** — every sweep dimension the benchmark iterates over, with the
  concrete set of values each takes. Format: one bullet per axis. Examples:
  - `dtype ∈ {float, half, __nv_bfloat16}`
  - `N ∈ {256, 1024, 4096}`
  - `block_size ∈ {64, 128, 256}`

  If the benchmark takes no axes (single measurement per run), say so
  explicitly: `Axes: none`.
- **Aggregation** — if a run produces multiple measurements (one per point in
  the axis product), how to reduce them to a single scalar for this
  benchmark. E.g. `min` for bandwidth (worst case wins), `max` for latency,
  `geomean` for a mix of shapes. For a no-axes benchmark, say `N/A`.

If the benchmarks have an obvious relationship — one wraps another, one is a
microbench of an operation that's part of another, etc. — note it under a
final "Relationships" paragraph. Otherwise treat them as independent targets.

### Cross-benchmark aggregation

If there's more than one benchmark, also specify how their per-benchmark
metrics combine into a single scalar that drives the keep-or-revert decision
when the user targets multiple benchmarks at once. Default if none is
specified: geometric mean of per-benchmark speedup versus baseline.

### Timeouts

Reasonable wall-time limits for:

- A full build from clean (e.g. "~2 minutes").
- An incremental rebuild after a source change (e.g. "~15 seconds").
- A single benchmark run (e.g. "~30 seconds").

Base these on observed times during discovery, with some margin.

### GPU and environment

- GPU model and compute capability.
- CUDA toolkit version.
- Any notable environment details (driver version, OS, etc.).

## Operating rules

**ACT AUTONOMOUSLY.** Do not ask the user for permission before installing packages, running builds, or executing benchmarks. If a dependency is missing, install it. If a build flag looks wrong, fix it. If something needs `sudo`, use it. The user expects you to solve problems, not ask about them.

**Only stop for genuine ambiguity.** If the project has multiple benchmark harnesses and you cannot determine which one to use, or if you cannot tell which files are in scope for optimization, ask. Otherwise, make a reasonable choice and move on. You can always revise `project-layout.md` later.

**Confirm at the end.** Once `project-layout.md` is written, present a brief summary of what you found (editable vs read-only files, metric, baseline result, any concerns) so the user can sanity-check before optimization begins.
