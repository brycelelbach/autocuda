# AutoCUDA: CUDA Kernel Optimizer

Autonomous CUDA kernel optimizer — analogous to
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) but targeting
GPU kernel performance instead of language model optimization.

## Overview

| File | Purpose | Editable? |
|------|---------|-----------|
| `kernel.cuh` | Kernel + launch config + metric declaration | **YES — only this** |
| `bench.cu` | Fixed nvbench harness | NO |
| `CMakeLists.txt` | Build system | NO |
| `autocuda.py` | Autonomous Claude API loop | NO |
| `program.md` | This file | Reference |
| `results.tsv` | Experiment log | Written by agent |

## Metrics

`FLOPS_PER_ELEMENT` does **not** choose what you optimise — it only declares how
many floating-point operations you count **per float** so `bench.cu` can report
FLOP/s (`N × NUM_FLOATS / mean GPU time`) when `N > 0`. A kernel can be
memory-bound and still have `FLOPS_PER_ELEMENT > 0`.

The optimisation target is chosen when running `autocuda.py`:

| `--metric` | What is optimised | Notes |
|------------|-------------------|--------|
| `bandwidth` (default) | **GlobalMem BW (GiB/s)** | Higher is better |
| `flops` | **FLOP/s (GFLOP/s)** | Higher is better; requires `FLOPS_PER_ELEMENT > 0` |
| `time` | **Mean GPU time (ms)** | Lower is better |

nvbench output includes timing and bandwidth; the FLOP/s summary appears when
`FLOPS_PER_ELEMENT > 0`.

## Kernel interface contract

`bench.cu` always launches the kernel as:

```cuda
kernel<<<compute_grid_size(NUM_FLOATS), BLOCK_SIZE, 0, stream>>>
       (src, dst, NUM_FLOATS)
```

where `NUM_FLOATS = 256 * 1024 * 1024 / sizeof(float) = 67,108,864`.

`kernel.cuh` **must** define:
- `static constexpr int BLOCK_SIZE`
- `static constexpr std::size_t FLOPS_PER_ELEMENT`
- `__global__ void kernel(const float*, float*, std::size_t)`
- `inline int compute_grid_size(std::size_t num_floats)`

The kernel may reinterpret the `float*` pointers internally (e.g. as `float4*`).

---

## Example: bandwidth kernel (default)

```cuda
// FLOPS_PER_ELEMENT = 0 → no FLOP/s summary (optimise bandwidth or time via --metric)
static constexpr std::size_t FLOPS_PER_ELEMENT = 0;
static constexpr int BLOCK_SIZE = 256;

__global__ void kernel(const float* __restrict__ src,
                       float* __restrict__       dst,
                       std::size_t               n)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                     i < n;
                     i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

inline int compute_grid_size(std::size_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
```

## Example: compute kernel (FLOP/s)

```cuda
// Each element performs 2*N_ITER FMAs → FLOPS_PER_ELEMENT = 2*N_ITER
static constexpr int N_ITER = 100;
static constexpr std::size_t FLOPS_PER_ELEMENT = 2 * N_ITER;
static constexpr int BLOCK_SIZE = 256;

__global__ void kernel(const float* __restrict__ src,
                       float* __restrict__       dst,
                       std::size_t               n)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                     i < n;
                     i += gridDim.x * blockDim.x) {
        float v = src[i];
        #pragma unroll
        for (int j = 0; j < N_ITER; ++j)
            v = __fmaf_rn(v, 1.00001f, 0.00001f);  // 2 FLOPs each
        dst[i] = v;
    }
}

inline int compute_grid_size(std::size_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
```

---

## Autonomous mode (Claude API loop)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python autocuda.py --metric bandwidth --iterations 30
# or: --metric flops   /   --metric time
```

Each iteration:
1. Reads the current `kernel.cuh` and `results.tsv`.
2. Sends both to Claude and asks for one improvement.
3. Applies the suggestion, rebuilds, and benchmarks.
4. Keeps the change if the primary metric improved; reverts otherwise.
5. Logs the result to `results.tsv`.

---

## Interactive mode (Claude Code session)

Open this directory in Claude Code and follow these instructions.

### Setup

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Benchmark

```bash
./build/bench
```

Look at `GlobalMem BW`, `GPU Time`, and (if `FLOPS_PER_ELEMENT > 0`) `FLOP/s` in the output.

### Workflow

1. Read the current `kernel.cuh` and the `results.tsv` log.
2. Choose one optimisation to try (see ideas below).
3. Edit `kernel.cuh`.
4. Run `cmake --build build --parallel && ./build/bench`.
5. If the chosen metric improved (higher BW/FLOP/s, or lower GPU time): keep the change, log it as `improved`.
6. If it regressed: revert `kernel.cuh`, log as `regressed`.
7. Append a row to `results.tsv`:
   ```
   <ISO timestamp>\t<metric_value>\t<unit>\t<status>\t<description>
   ```
8. Repeat.

### Bandwidth optimisation ideas

1. **Vectorised 128-bit loads** — cast `float*` to `float4*`; each thread copies
   4 floats per load/store. Adjust `compute_grid_size` to divide by 4.
2. **`__ldg()` cache hint** — `__ldg(&src[i])` routes through the read-only cache.
3. **BLOCK_SIZE tuning** — try 64, 128, 256, 512.
4. **Multiple elements per thread** — wider grid-stride loop.
5. **Loop unrolling** — `#pragma unroll 4`.

### Compute (FLOP/s) optimisation ideas

1. **Increase arithmetic intensity** — more FMAs per element, update `FLOPS_PER_ELEMENT`.
2. **Independent FMA chains** — multiple independent accumulators per thread to
   hide FP latency and saturate execution units.
3. **Half-precision** — `__half` or `__half2` can double throughput on modern GPUs.
4. **Warp-level reductions** — `__shfl_xor_sync` for efficient cross-lane ops.

---

## Results file format

`results.tsv` columns:

| column | example |
|--------|---------|
| `timestamp` | `2026-04-05T14:32:01` |
| `metric_value` | `487.3412` |
| `unit` | `GiB/s` or `GFLOP/s` |
| `status` | `baseline` / `improved` / `regressed` / `build_error` / `runtime_error` |
| `description` | `Added float4 vectorised loads` |
