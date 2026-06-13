# Ternary Thread Block — GPU Thread Block Configuration for Ternary Kernels

**Ternary Thread Block** provides abstractions for configuring GPU thread blocks, warp allocations, and occupancy calculations specifically for ternary (three-valued logic) GPU kernels. It validates configurations against hardware limits, computes occupancy, and generates launch parameters optimized for ternary workload patterns.

## Why It Matters

GPU performance depends critically on thread block configuration: too many threads per block wastes registers, too few underutilizes the multiprocessor. For ternary kernels, the optimal configuration differs from standard floating-point kernels because ternary operations use fewer registers (2 bits vs 32 bits per value) and less shared memory. This crate computes the optimal block size, shared memory allocation, and occupancy for ternary kernels given hardware constraints, ensuring that ternary GPU kernels achieve maximum throughput.

## How It Works

### Thread Block Configuration

A `ThreadBlock` specifies:
- `dim_x`, `dim_y`, `dim_z`: Thread dimensions
- `shared_mem_per_block`: Shared memory allocation in bytes

Total threads = dim_x × dim_y × dim_z. The crate provides constructors for 1D, 2D, and 3D blocks.

### Hardware Validation

`validate(max_threads_per_block, max_shared_mem)` checks:
1. No dimension is zero
2. Total threads ≤ hardware limit (typically 1024)
3. Shared memory ≤ hardware limit (typically 48KB)

Returns `BlockError` with specific violation details. O(1).

### Occupancy Calculation

Occupancy = active_warps / max_warps_per_SM. For ternary kernels:
- Each thread uses fewer registers (ternary values pack 16/u32 vs 1 float/u32)
- Shared memory usage is lower (2-bit vs 32-bit data)
- This allows higher occupancy — more concurrent warps per streaming multiprocessor

The `occupancy()` function computes the ratio given block configuration, registers per thread, and SM limits. O(1).

### Grid Configuration

`Grid::from_problem(problem_size, block_size)` computes the grid dimensions needed to cover a problem of given size with given block dimensions. Handles 1D, 2D, and 3D problems with ceiling division. O(1).

### Warps and Occupancy

Each `WarpInfo` tracks:
- Thread assignments
- Divergence patterns (important for ternary branches where 3-way divergence is possible)
- Synchronization barriers

## Quick Start

```rust
use ternary_thread_block::ThreadBlock;

// Configure a 1D block for ternary matmul
let block = ThreadBlock::new_1d(256, 4096); // 256 threads, 4KB shared mem

// Validate against RTX 4050 limits
block.validate(1024, 49152).expect("valid configuration");
assert_eq!(block.total_threads(), 256);
```

```bash
cargo add ternary-thread-block
```

## API

| Type / Function | Description |
|---|---|
| `ThreadBlock` | `{ dim_x, dim_y, dim_z, shared_mem_per_block }` |
| `ThreadBlock::new_1d/2d/3d()` | Dimension constructors |
| `validate(max_threads, max_smem)` | Hardware constraint check |
| `total_threads()` | dim_x × dim_y × dim_z |
| `BlockError` | ZeroDimension, TooManyThreads, TooMuchSharedMem |

## Architecture Notes

Thread block configuration optimizes ternary GPU kernel launches in **SuperInstance**. Ternary kernels achieve higher occupancy than FP32 kernels due to lower register and shared memory usage. The γ + η = C conservation manifests in the occupancy trade-off: more active threads (γ = throughput) requires more shared memory (η = resource cost), bounded by SM capacity C. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Kirk, David & Hwu, Wen-mei. *Programming Massively Parallel Processors*, 4th ed., Morgan Kaufmann, 2022.
| NVIDIA. *CUDA C++ Programming Guide*, v12, 2024 — occupancy calculation.
| Hennessy, John & Patterson, David. *Computer Architecture*, 6th ed., 2017.

## License

MIT
