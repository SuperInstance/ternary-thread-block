# ternary-thread-block

Thread block scheduling for **ternary GPU kernels**. Provides abstractions for managing thread block dimensions, warp allocations, streaming multiprocessor (SM) occupancy, and memory alignment — all tuned for kernels operating on three-valued (ternary) data where 16 trits pack into a single `u32`.

## Why It Matters

Ternary GPU kernels (e.g., BitNet matrix multiplications, ternary convolutions) have unique hardware requirements:

1. **Packed memory access** — 16 trits per `u32` means optimal thread counts must be multiples of 16
2. **Shared memory alignment** — ternary packed buffers require 4-byte aligned access
3. **Warp efficiency** — 32-thread warps should process whole `u32` packs (2 packs per warp)
4. **Occupancy calculation** — shared memory per block determines how many blocks fit per SM

This crate computes all of these, validates against hardware limits, and suggests optimal configurations.

## How It Works

### Thread Block Configuration

A `ThreadBlock` specifies 3D dimensions and shared memory:

```
ThreadBlock {
    dim_x, dim_y, dim_z: u32,    // threads per dimension
    shared_mem_per_block: u32,    // bytes
}
total_threads = dim_x · dim_y · dim_z
```

### Validation

Checks against GPU hardware constraints:

```
if total_threads > max_threads_per_block (typically 1024):
    → Error::TooManyThreads
if shared_mem_per_block > max_shared_mem (typically 48KB):
    → Error::TooMuchSharedMem
if any dim == 0:
    → Error::ZeroDimension
```

**Complexity:** O(1).

### Warp Calculation

A warp is 32 threads. The number of warps per block:

```
warps_per_block = ⌈total_threads / 32⌉
```

For ternary kernels, the optimal block size is a multiple of 32 (one or more full warps) and a multiple of 16 (the pack width). Since `gcd(32, 16) = 16`, any multiple of 32 satisfies both.

### Occupancy Calculation

SM occupancy = how many blocks can run simultaneously on one SM:

```
blocks_per_sm = min(
    ⌊max_threads_per_sm / total_threads⌋,
    ⌊max_shared_mem_per_sm / shared_mem_per_block⌋,
    max_blocks_per_sm
)
occupancy = blocks_per_sm · warps_per_block / max_warps_per_sm
```

For NVIDIA SMs (typical):
- `max_threads_per_sm = 2048`
- `max_warps_per_sm = 64`
- `max_blocks_per_sm = 32`

**Complexity:** O(1).

### Ternary Grid Configuration

For a ternary matrix of *M × N* elements (packed as `M · N / 16` `u32` words), the grid configuration:

```
threads_per_block = 256 (8 warps, 16 packs)
blocks_needed = ⌈total_packs / (threads_per_block / 16)⌉
              = ⌈total_packs / 16⌉
grid_x = ⌈blocks_needed⌉
grid_y = 1
grid_z = 1
```

This ensures each thread processes exactly one `u32` pack (16 trits), maximizing memory coalescing.

### Memory Alignment

Shared memory for ternary buffers must be 4-byte aligned (one `u32`). The `shared_mem_per_block` field tracks this:

```
required_shared = num_packs_per_block · 4  // bytes
```

The validator checks `required_shared ≤ max_shared_mem`.

## Quick Start

```rust
use ternary_thread_block::{ThreadBlock, GridConfig, OccupancyCalculator};

// Configure a 256-thread block for ternary matmul
let block = ThreadBlock::new_1d(256, shared_mem: 4096);
block.validate(max_threads: 1024, max_shared: 48_000).unwrap();

assert_eq!(block.total_threads(), 256);

// Compute occupancy on a typical SM
let calc = OccupancyCalculator::new(max_threads_per_sm: 2048, max_blocks_per_sm: 32, max_shared_per_sm: 48_000);
let occupancy = calc.compute(&block);
assert!(occupancy.active_warps > 0);
```

## API

| Type | Key Methods |
|------|-------------|
| `ThreadBlock` | `new_1d(x, smem)`, `new_2d(x, y, smem)`, `new_3d(x, y, z, smem)`, `total_threads()`, `validate()` |
| `GridConfig` | `for_ternary_matrix(m, n, threads_per_block)`, `blocks_needed()` |
| `OccupancyCalculator` | `new(...)`, `compute(block)`, `max_active_blocks(block)` |
| `BlockError` | `ZeroDimension`, `TooManyThreads`, `TooMuchSharedMem` |

## Architecture Notes

The **γ + η = C** invariant: *generation* (γ) is the thread block configuration producing parallel work, *entropy* (η) is the occupancy loss — the gap between achieved warps per SM and the maximum (64). *Conservation* (C) is the hardware resource invariant: `blocks_per_sm · threads_per_block ≤ max_threads_per_sm` and `blocks_per_sm · shared_mem_per_block ≤ max_shared_mem_per_sm`. The occupancy calculator explicitly evaluates these conservation constraints to determine how many blocks can coexist, and the ternary pack alignment (16 trits/u32) ensures that thread blocks process data at the natural granularity of the ternary representation.

## References

- **CUDA occupancy guide:** NVIDIA, "CUDA C++ Programming Guide" §7 (Hardware Implementation)
- **Warp-level programming:** Kirk, D. & Hwu, W. *Programming Massively Parallel Processors* (2016), Chapter 5
- **Ternary weight packing:** Alemdar, H. et al. "Ternary Weight Networks" (2017), §3.1
- **Shared memory optimization:** Harris, M. "Optimizing CUDA" (2020), GPU Technology Conference

## License

MIT
