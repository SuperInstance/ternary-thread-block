# ternary-thread-block

Thread block scheduling for ternary GPU kernels.

## Overview

`ternary-thread-block` provides abstractions for managing GPU thread blocks, warp allocations, and occupancy calculations tailored for ternary (three-valued logic) GPU kernels. It helps configure and schedule thread blocks across streaming multiprocessors (SMs) efficiently, ensuring optimal resource utilization for ternary compute workloads.

## Features

- **ThreadBlock** — Define 1D, 2D, or 3D thread block configurations with shared memory per block and hardware validation.
- **BlockScheduler** — Assign thread blocks to streaming multiprocessors using round-robin or greedy strategies, respecting thread and block limits per SM.
- **WarpAllocation** — Compute warp decomposition within a block, track full and partial warps, and query per-warp thread ranges.
- **Occupancy Calculator** — Calculate SM occupancy considering thread count, warp limits, block limits, and shared memory constraints.
- **BlockConfig Builder** — Fluent builder API for constructing and validating thread block configurations.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-thread-block = { git = "https://github.com/SuperInstance/ternary-thread-block" }
```

### Basic Example

```rust
use ternary_thread_block::*;

// Build a 1D thread block with 256 threads and 4KB shared memory
let block = BlockConfig::new()
    .dim_x(256)
    .shared_mem(4096)
    .build_validated(1024, 49152)
    .unwrap();

// Allocate warps
let warp_alloc = WarpAllocation::from_block(&block, 32);
println!("Warps: {}, full: {}, partial: {}",
    warp_alloc.num_warps,
    warp_alloc.full_warp_count(),
    warp_alloc.has_partial_warp()
);

// Schedule across 10 SMs
let scheduler = BlockScheduler::new(10, 32, 2048);
let assignments = scheduler.schedule_round_robin(100, &block);
println!("All blocks scheduled: {}", scheduler.all_blocks_scheduled(&assignments, 100));

// Calculate occupancy
let occupancy = calculate_occupancy(&block, &scheduler, 32, 64, 49152);
println!("Occupancy: {:.1}%", occupancy.occupancy_ratio * 100.0);
```

## Architecture

### ThreadBlock

A `ThreadBlock` encapsulates 3D thread dimensions and shared memory allocation. It validates against GPU hardware constraints (max threads per block, max shared memory) and computes total thread count.

### WarpAllocation

Breaks a thread block into warps based on the GPU warp size (typically 32). Tracks which thread indices belong to each warp, identifies partial warps, and provides per-warp thread ranges for kernel scheduling.

### BlockScheduler

Two scheduling strategies:
- **Round-robin** — Distributes blocks evenly across SMs, cycling through each SM one block at a time.
- **Greedy** — Fills each SM to capacity before moving to the next.

Both strategies respect per-SM limits on thread count, block count, and shared memory.

### Occupancy Calculation

The `calculate_occupancy` function considers four limiting factors:
1. Maximum threads per SM
2. Maximum warps per SM
3. Maximum blocks per SM
4. Shared memory per SM

Returns the active blocks per SM, warps per SM, and occupancy ratio (0.0–1.0).

## Validation

All configurations are validated against hardware limits. `BlockError` variants cover zero dimensions, exceeding thread limits, and exceeding shared memory limits.

## License

MIT
