# autocuda

Autonomous CUDA kernel optimizer - inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) but targeting
GPU kernel performance instead of language-model training.

The idea: give an AI agent a simple CUDA kernel and a fixed benchmark harness,
then let it experiment autonomously. It modifies the kernel, benchmarks it,
checks if the result improved, keeps or discards, and repeats. You come back to
a log of experiments and (hopefully) a faster kernel.

## How it works

The repo is deliberately kept small and only has three files that matter:

- **`bench.cu`** - fixed [nvbench](https://github.com/NVIDIA/nvbench) harness.
  Not modified.
- **`kernel.cuh`** - the single file the agent edits. Contains the kernel
  template, block size, grid-size computation, and explicit template
  instantiations. Everything is fair game: vectorisation, memory access
  patterns, block size, loop structure, type-specific specialisations, etc.
  **This file is edited and iterated on by the agent.**
- **`autocuda.py`** - the outer loop. Reads the current kernel and experiment
  history, asks Claude for one improvement, applies it, benchmarks, and keeps or
  reverts. **This file is not edited by the agent.**

The optimisation target is configurable:

| `--metric` | What is optimised | Unit | Direction |
|------------|-------------------|------|-----------|
| `bandwidth` (default) | Global memory bandwidth | GiB/s | Higher is better |
| `time` | Mean GPU time | ms | Lower is better |
| `flops` | Throughput (via `add_element_count`) | GFLOP/s | Higher is better |

When the harness produces multiple measurements, `--aggregate` controls how they
are combined into a single scalar (defaults: `min` for bandwidth, `max` for
time).

## Quick start

**Requirements:** a single NVIDIA GPU, CMake 3.30+, a CUDA toolkit, Python
3.10+, and an [Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Build the benchmark
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Run a manual baseline to verify setup
./build/bench

# 3. Run the autonomous optimizer
export ANTHROPIC_API_KEY=sk-ant-...
python autocuda.py --metric bandwidth --iterations 30
```

## Running the agent

### Autonomous mode (API loop)

`autocuda.py` drives the full experiment loop programmatically. Each iteration
it asks Claude for one kernel change, applies it, benchmarks, and keeps or
reverts.

```bash
python autocuda.py --metric bandwidth --iterations 30
python autocuda.py --metric time --iterations 20 --bench-timeout 30
```

The agent's optimisation strategy is defined in
`cuda-kernel-optimization-idea-skill.md`.

### Interactive mode (Claude Code / Cursor)

Open this directory in your agent-enabled editor and point it at
`cuda-kernel-optimization-iteration-skill.md`. The skill instructs the agent to
run the experiment loop autonomously - you can leave it running and come back to
results.

## Project structure

```
bench.cu            - fixed nvbench harness (do not modify)
kernel.cuh          - kernel source (agent modifies this)
CMakeLists.txt      - build system (do not modify)
autocuda.py         - autonomous API-loop driver
results.csv         - experiment log (written by agent / script)

cuda-kernel-optimization-idea-skill.md       - skill: generate one kernel optimisation
cuda-kernel-optimization-iteration-skill.md  - skill: autonomous experiment loop
```

## Design choices

- **Single file to modify.** The agent only touches `kernel.cuh`. This keeps
  scope manageable and diffs reviewable.
- **Fixed benchmark harness.** `bench.cu` and `CMakeLists.txt` are never
  modified. The benchmark is the ground truth - consistent and reproducible.
- **Multiple benchmark states.** The harness can sweep across types, sizes, or
  other axes. This forces the agent to find optimisations that generalise.
- **Self-contained.** One GPU, one kernel file, one metric. No distributed
  builds, no complex configs.
