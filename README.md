# autocuda

Autonomous CUDA kernel optimizer - inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) but targeting
GPU kernel performance instead of language-model training.

The idea: give an AI agent a simple CUDA kernel and a fixed benchmark harness,
then let it experiment autonomously. It modifies the kernel, benchmarks it,
checks if the result improved, keeps or discards, and repeats. You come back to
a log of experiments and (hopefully) a faster kernel.

## How it works

The repo ships several example kernels under `kernels/`, each with its own
fixed benchmark harness and a naive starter kernel:

| Kernel | Description | Key metric |
|--------|-------------|------------|
| `memcpy` | Vectorised device-to-device copy | memory-bandwidth |
| `stencil` | 5-point heat equation with constant boundary conditions | memory-bandwidth |
| `matmul` | Dense matrix multiplication C = A * B | compute-bandwidth |
| `sigmoid` | PyTorch-style element-wise sigmoid | memory-bandwidth |

Each kernel directory has two files:

- **`bench.cu`** - fixed [nvbench](https://github.com/NVIDIA/nvbench) harness.
  Not modified.
- **`kernel.cuh`** - the single file the agent edits. Contains the kernel
  template, block size, grid-size computation, and explicit template
  instantiations. Everything is fair game: vectorisation, memory access
  patterns, block size, loop structure, type-specific specialisations, etc.
  **This file is edited and iterated on by the agent.**

The outer loop is driven by:

- **`autocuda.py`** - reads the current kernel and experiment
  history, asks Claude for one improvement, applies it, benchmarks, and keeps or
  reverts. **This file is not edited by the agent.** Use `--kernel <name>` to
  select which kernel to optimize.

The optimization target is configurable:

| `--metric` | What is optimized | Unit | Direction |
|------------|-------------------|------|-----------|
| `memory-bandwidth` (default) | Global memory bandwidth | GiB/s | Higher is better |
| `compute-bandwidth` | Throughput (via `add_element_count`) | GFLOP/s | Higher is better |
| `time` | Mean GPU time | ms | Lower is better |

When the harness produces multiple measurements, `--aggregate` controls how they
are combined into a single scalar (defaults: `min` for memory-bandwidth, `max` for
time).

## Quick start

**Requirements:** a single NVIDIA GPU, CMake 3.30+, a CUDA toolkit, Python
3.10+, and an [Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Build all benchmarks
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Run a manual baseline to verify setup
./build/bench_memcpy

# 3. Run the autonomous optimizer on a specific kernel
export ANTHROPIC_API_KEY=sk-ant-...
python autocuda.py --kernel memcpy --metric memory-bandwidth --iterations 30
python autocuda.py --kernel matmul --metric compute-bandwidth --iterations 30
```

## Running the agent

### Autonomous mode (API loop)

`autocuda.py` drives the full experiment loop programmatically. Each iteration
it asks Claude for one kernel change, applies it, benchmarks, and keeps or
reverts.

```bash
python autocuda.py --kernel memcpy  --metric memory-bandwidth --iterations 30
python autocuda.py --kernel matmul  --metric compute-bandwidth --iterations 20
python autocuda.py --kernel sigmoid --metric memory-bandwidth --iterations 20
python autocuda.py --kernel stencil --metric memory-bandwidth --iterations 20
```

The agent's optimization strategy is defined in
`cuda-kernel-optimization-idea-skill.md`.

### Interactive mode (Claude Code / Cursor)

Open this directory in your agent-enabled editor and point it at
`cuda-kernel-optimization-iteration-skill.md`. The skill instructs the agent to
run the experiment loop autonomously - you can leave it running and come back to
results.

## Project structure

```
kernels/
  memcpy/            - vectorised device-to-device copy
    bench.cu         - fixed nvbench harness (do not modify)
    kernel.cuh       - kernel source (agent modifies this)
  stencil/           - 5-point heat equation stencil
    bench.cu
    kernel.cuh
  matmul/            - dense matrix multiplication
    bench.cu
    kernel.cuh
  sigmoid/           - PyTorch-style sigmoid operator
    bench.cu
    kernel.cuh
CMakeLists.txt       - build system (do not modify)
autocuda.py          - autonomous API-loop driver (--kernel selects workload)
results.csv          - experiment log (written by agent / script)

cuda-kernel-optimization-idea-skill.md       - skill: generate one kernel optimization
cuda-kernel-optimization-iteration-skill.md  - skill: autonomous experiment loop
```

## Design choices

- **Single file to modify.** The agent only touches `kernel.cuh` for the
  selected workload. This keeps scope manageable and diffs reviewable.
- **Fixed benchmark harness.** `bench.cu` and `CMakeLists.txt` are never
  modified. The benchmark is the ground truth - consistent and reproducible.
- **Multiple kernels.** The repo ships several example workloads (memcpy,
  stencil, matmul, sigmoid) to evaluate how well the optimizer generalises
  across different problem types.
- **Multiple benchmark states.** Each harness can sweep across types, sizes, or
  other axes. This forces the agent to find optimizations that generalise.
- **Self-contained.** One GPU, one kernel file, one metric. No distributed
  builds, no complex configs.
