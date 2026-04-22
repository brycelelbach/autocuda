# autocuda

Autonomous CUDA optimizer - inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) but targeting
GPU kernel performance instead of language-model training.

The idea: give an agent a CUDA project with a fixed benchmark harness, then let
it experiment autonomously. It modifies source, validates correctness, benchmarks
it, checks if the result improved, keeps or discards, and repeats. You come back
to a log of trials and (hopefully) a faster codebase.

## How it works

autocuda ships two Claude Code skills:

- **`autocuda:discover`** - explores an unfamiliar CUDA project, identifies
  editable vs read-only files, verifies the build and benchmark run, and
  writes a `project-layout.md` that pins down the ground truth for the
  optimization loop.
- **`autocuda:optimize`** - reads `project-layout.md`, then runs the trial
  loop: modify source, build, validate, benchmark, keep-or-revert, log,
  repeat. Commits every winning change on an `experiment/<tag>` branch so
  the history is reviewable.

The repo also ships several example kernels under `kernels/` that can be used
as targets for the skills:

| Kernel | Description | Key metric |
|--------|-------------|------------|
| `memcpy` | Vectorised device-to-device copy | memory-bandwidth |
| `stencil` | 5-point heat equation with constant boundary conditions | memory-bandwidth |
| `matmul` | Dense matrix multiplication C = A * B | compute-bandwidth |
| `sigmoid` | PyTorch-style element-wise sigmoid | memory-bandwidth |

Each kernel directory has two files:

- **`bench.cu`** - fixed [nvbench](https://github.com/NVIDIA/nvbench) harness.
  Read-only.
- **`kernel.cuh`** - the file the agent edits. Contains the kernel template,
  block size, grid-size computation, and explicit template instantiations.
  Everything is fair game: vectorisation, memory access patterns, block size,
  loop structure, type-specific specialisations, etc.

## Install

**Requirements:** a single NVIDIA GPU, CMake 3.30+, a CUDA toolkit, and
[Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Install the plugin via Claude Code's plugin marketplace:

```
/plugin marketplace add brycelelbach/autocuda
/plugin install autocuda@brycelelbach-autocuda
```

Invoke via `/autocuda:discover` and `/autocuda:optimize`.

For unattended setup (Claude Code install + marketplace registration in one
shot), run `bootstrap.bash`:

```bash
curl -fsSL https://raw.githubusercontent.com/brycelelbach/autocuda/main/bootstrap.bash | bash
```

## Quick start

To try the skills against the example kernels bundled with this repo:

```bash
# 1. Build all benchmarks
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Run a manual baseline to verify setup
./build/bench_memcpy
```

Then open this directory in Claude Code, run the `autocuda:discover` skill to
produce `project-layout.md`, then hand off to the `autocuda:optimize` skill to
run the trial loop.

## Project structure

```
.claude-plugin/
  marketplace.json           - marketplace manifest (consumed by /plugin marketplace add)
plugins/
  autocuda/
    .claude-plugin/
      plugin.json            - plugin manifest
    skills/
      discover/              - discover project structure, produce project-layout.md
      optimize/              - run the autonomous optimization trial loop

kernels/                     - example CUDA workloads used as optimization targets
  memcpy/
    bench.cu                 - fixed nvbench harness (do not modify)
    kernel.cuh               - kernel source (agent modifies this)
  stencil/                   - 5-point heat equation stencil
  matmul/                    - dense matrix multiplication
  sigmoid/                   - PyTorch-style sigmoid operator
CMakeLists.txt               - build system for the example kernels
```

## Design choices

- **Discovery then optimization.** A separate discovery pass produces the
  ground-truth file (`project-layout.md`), so the optimization loop doesn't
  have to re-derive project structure on every run.
- **Fixed benchmark harness.** `bench.cu` and `CMakeLists.txt` are never
  modified. The benchmark is the ground truth - consistent and reproducible.
- **Multiple kernels.** The repo ships several example workloads (memcpy,
  stencil, matmul, sigmoid) to evaluate how well the optimizer generalises
  across different problem types.
- **Multiple benchmark states.** Each harness can sweep across types, sizes, or
  other axes. This forces the agent to find optimizations that generalise.
- **Self-contained.** One GPU, one repo, one metric at a time. No distributed
  builds, no complex configs.
