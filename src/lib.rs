//! # ternary-thread-block
//!
//! Thread block scheduling for ternary GPU kernels.
//!
//! This crate provides abstractions for managing GPU thread blocks,
//! warp allocations, and occupancy calculations tailored for ternary
//! (three-valued logic) GPU kernels. It helps configure and schedule
//! thread blocks across streaming multiprocessors (SMs) efficiently.

use std::fmt;

/// Represents a thread block configuration with dimensions and shared memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadBlock {
    /// Number of threads in the x dimension.
    pub dim_x: u32,
    /// Number of threads in the y dimension.
    pub dim_y: u32,
    /// Number of threads in the z dimension.
    pub dim_z: u32,
    /// Shared memory per block in bytes.
    pub shared_mem_per_block: u32,
}

impl ThreadBlock {
    /// Create a new 1D thread block.
    pub fn new_1d(threads_x: u32, shared_mem: u32) -> Self {
        Self {
            dim_x: threads_x,
            dim_y: 1,
            dim_z: 1,
            shared_mem_per_block: shared_mem,
        }
    }

    /// Create a new 2D thread block.
    pub fn new_2d(threads_x: u32, threads_y: u32, shared_mem: u32) -> Self {
        Self {
            dim_x: threads_x,
            dim_y: threads_y,
            dim_z: 1,
            shared_mem_per_block: shared_mem,
        }
    }

    /// Create a new 3D thread block.
    pub fn new_3d(threads_x: u32, threads_y: u32, threads_z: u32, shared_mem: u32) -> Self {
        Self {
            dim_x: threads_x,
            dim_y: threads_y,
            dim_z: threads_z,
            shared_mem_per_block: shared_mem,
        }
    }

    /// Total number of threads in the block.
    pub fn total_threads(&self) -> u32 {
        self.dim_x * self.dim_y * self.dim_z
    }

    /// Validate that the block conforms to GPU hardware limits.
    pub fn validate(&self, max_threads_per_block: u32, max_shared_mem: u32) -> Result<(), BlockError> {
        if self.dim_x == 0 || self.dim_y == 0 || self.dim_z == 0 {
            return Err(BlockError::ZeroDimension);
        }
        let total = self.total_threads();
        if total > max_threads_per_block {
            return Err(BlockError::TooManyThreads {
                requested: total,
                maximum: max_threads_per_block,
            });
        }
        if self.shared_mem_per_block > max_shared_mem {
            return Err(BlockError::TooMuchSharedMem {
                requested: self.shared_mem_per_block,
                maximum: max_shared_mem,
            });
        }
        Ok(())
    }
}

/// Errors that can occur during block configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockError {
    /// A block dimension was set to zero.
    ZeroDimension,
    /// Thread count exceeds hardware maximum.
    TooManyThreads { requested: u32, maximum: u32 },
    /// Shared memory exceeds hardware maximum.
    TooMuchSharedMem { requested: u32, maximum: u32 },
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::ZeroDimension => write!(f, "block dimension cannot be zero"),
            BlockError::TooManyThreads { requested, maximum } => {
                write!(f, "requested {} threads but maximum is {}", requested, maximum)
            }
            BlockError::TooMuchSharedMem { requested, maximum } => {
                write!(f, "requested {} bytes shared mem but maximum is {}", requested, maximum)
            }
            BlockError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
        }
    }
}

impl std::error::Error for BlockError {}

/// Warp allocation within a thread block.
#[derive(Debug, Clone)]
pub struct WarpAllocation {
    /// Warp size (typically 32 on NVIDIA GPUs).
    pub warp_size: u32,
    /// Total number of warps in this block.
    pub num_warps: u32,
    /// Warp assignments: which warp handles which portion of work.
    pub warp_ranges: Vec<(u32, u32)>,
}

impl WarpAllocation {
    /// Allocate warps for a given thread block.
    pub fn from_block(block: &ThreadBlock, warp_size: u32) -> Self {
        let total_threads = block.total_threads();
        let num_warps = (total_threads + warp_size - 1) / warp_size;
        let mut warp_ranges = Vec::with_capacity(num_warps as usize);
        for w in 0..num_warps {
            let start = w * warp_size;
            let end = std::cmp::min(start + warp_size, total_threads);
            warp_ranges.push((start, end));
        }
        Self {
            warp_size,
            num_warps,
            warp_ranges,
        }
    }

    /// Get the range of thread indices handled by a specific warp.
    pub fn warp_range(&self, warp_idx: u32) -> Option<(u32, u32)> {
        self.warp_ranges.get(warp_idx as usize).copied()
    }

    /// Threads assigned to a specific warp.
    pub fn threads_for_warp(&self, warp_idx: u32) -> u32 {
        self.warp_range(warp_idx)
            .map(|(s, e)| e - s)
            .unwrap_or(0)
    }

    /// Full warps (warps with exactly warp_size threads).
    pub fn full_warp_count(&self) -> u32 {
        self.warp_ranges
            .iter()
            .filter(|&&(s, e)| e - s == self.warp_size)
            .count() as u32
    }

    /// Whether the last warp is partial (has fewer threads than warp_size).
    pub fn has_partial_warp(&self) -> bool {
        self.warp_ranges
            .last()
            .map(|&(s, e)| e - s < self.warp_size)
            .unwrap_or(false)
    }
}

/// Block assignment to a streaming multiprocessor (SM).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmAssignment {
    /// SM index.
    pub sm_id: u32,
    /// Blocks assigned to this SM.
    pub blocks: Vec<u32>,
}

impl SmAssignment {
    /// Number of blocks on this SM.
    pub fn block_count(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// Total threads on this SM.
    pub fn total_threads(&self, block: &ThreadBlock) -> u32 {
        self.blocks.len() as u32 * block.total_threads()
    }
}

/// Block scheduler that assigns thread blocks to SMs.
#[derive(Debug, Clone)]
pub struct BlockScheduler {
    /// Number of SMs available.
    pub num_sms: u32,
    /// Maximum blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,
}

impl BlockScheduler {
    /// Create a new scheduler with GPU parameters.
    pub fn new(num_sms: u32, max_blocks_per_sm: u32, max_threads_per_sm: u32) -> Self {
        Self {
            num_sms,
            max_blocks_per_sm,
            max_threads_per_sm,
        }
    }

    /// Schedule blocks in round-robin fashion across SMs.
    pub fn schedule_round_robin(&self, total_blocks: u32, block: &ThreadBlock) -> Vec<SmAssignment> {
        let threads_per_block = block.total_threads();
        let mut assignments: Vec<SmAssignment> = (0..self.num_sms)
            .map(|i| SmAssignment {
                sm_id: i,
                blocks: Vec::new(),
            })
            .collect();

        let mut block_idx: u32 = 0;
        loop {
            let mut placed_any = false;
            for sm in &mut assignments {
                if block_idx >= total_blocks {
                    break;
                }
                let current_threads = sm.blocks.len() as u32 * threads_per_block;
                if sm.blocks.len() as u32 >= self.max_blocks_per_sm {
                    continue;
                }
                if current_threads + threads_per_block > self.max_threads_per_sm {
                    continue;
                }
                sm.blocks.push(block_idx);
                block_idx += 1;
                placed_any = true;
            }
            if block_idx >= total_blocks || !placed_any {
                break;
            }
        }

        assignments
    }

    /// Schedule blocks greedily, filling each SM before moving to the next.
    pub fn schedule_greedy(&self, total_blocks: u32, block: &ThreadBlock) -> Vec<SmAssignment> {
        let threads_per_block = block.total_threads();
        let mut assignments: Vec<SmAssignment> = (0..self.num_sms)
            .map(|i| SmAssignment {
                sm_id: i,
                blocks: Vec::new(),
            })
            .collect();

        let mut block_idx: u32 = 0;
        for sm in &mut assignments {
            while block_idx < total_blocks {
                let current_threads = sm.blocks.len() as u32 * threads_per_block;
                if sm.blocks.len() as u32 >= self.max_blocks_per_sm {
                    break;
                }
                if current_threads + threads_per_block > self.max_threads_per_sm {
                    break;
                }
                sm.blocks.push(block_idx);
                block_idx += 1;
            }
        }

        assignments
    }

    /// Check if all blocks were scheduled.
    pub fn all_blocks_scheduled(&self, assignments: &[SmAssignment], total_blocks: u32) -> bool {
        let scheduled: u32 = assignments.iter().map(|a| a.blocks.len() as u32).sum();
        scheduled == total_blocks
    }

    /// Calculate how many blocks can fit on each SM given shared memory constraints.
    pub fn blocks_per_sm_shared_mem(&self, shared_mem_per_block: u32, total_shared_per_sm: u32) -> u32 {
        if shared_mem_per_block == 0 {
            return self.max_blocks_per_sm;
        }
        std::cmp::min(
            total_shared_per_sm / shared_mem_per_block,
            self.max_blocks_per_sm,
        )
    }
}

/// Occupancy information for a kernel launch configuration.
#[derive(Debug, Clone)]
pub struct OccupancyInfo {
    /// Number of warps per SM.
    pub warps_per_sm: u32,
    /// Maximum warps per SM on this hardware.
    pub max_warps_per_sm: u32,
    /// Occupancy ratio (0.0 to 1.0).
    pub occupancy_ratio: f64,
    /// Number of active blocks per SM.
    pub active_blocks_per_sm: u32,
}

/// Calculate occupancy for a given block configuration and hardware.
pub fn calculate_occupancy(
    block: &ThreadBlock,
    scheduler: &BlockScheduler,
    warp_size: u32,
    max_warps_per_sm: u32,
    shared_mem_per_sm: u32,
) -> OccupancyInfo {
    let threads_per_block = block.total_threads();
    let warps_per_block = (threads_per_block + warp_size - 1) / warp_size;

    // Limiting factors for active blocks per SM
    let blocks_by_threads = if threads_per_block > 0 {
        scheduler.max_threads_per_sm / threads_per_block
    } else {
        0
    };
    let blocks_by_warps = if warps_per_block > 0 {
        max_warps_per_sm / warps_per_block
    } else {
        0
    };
    let blocks_by_count = scheduler.max_blocks_per_sm;
    let blocks_by_shared = if block.shared_mem_per_block > 0 {
        shared_mem_per_sm / block.shared_mem_per_block
    } else {
        blocks_by_count
    };

    let active_blocks = blocks_by_threads
        .min(blocks_by_warps)
        .min(blocks_by_count)
        .min(blocks_by_shared);

    let warps_per_sm = active_blocks * warps_per_block;
    let occupancy_ratio = if max_warps_per_sm > 0 {
        warps_per_sm as f64 / max_warps_per_sm as f64
    } else {
        0.0
    };

    OccupancyInfo {
        warps_per_sm,
        max_warps_per_sm,
        occupancy_ratio,
        active_blocks_per_sm: active_blocks,
    }
}

/// Builder for thread block configurations.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    shared_mem: u32,
}

impl BlockConfig {
    /// Start building a 1D block configuration.
    pub fn new() -> Self {
        Self {
            dim_x: 256,
            dim_y: 1,
            dim_z: 1,
            shared_mem: 0,
        }
    }

    /// Set x dimension (thread count for 1D).
    pub fn dim_x(mut self, x: u32) -> Self {
        self.dim_x = x;
        self
    }

    /// Set y dimension.
    pub fn dim_y(mut self, y: u32) -> Self {
        self.dim_y = y;
        self
    }

    /// Set z dimension.
    pub fn dim_z(mut self, z: u32) -> Self {
        self.dim_z = z;
        self
    }

    /// Set shared memory per block in bytes.
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Build a thread block from this configuration.
    pub fn build(self) -> ThreadBlock {
        ThreadBlock {
            dim_x: self.dim_x,
            dim_y: self.dim_y,
            dim_z: self.dim_z,
            shared_mem_per_block: self.shared_mem,
        }
    }

    /// Build and validate the block against hardware limits.
    pub fn build_validated(self, max_threads: u32, max_shared: u32) -> Result<ThreadBlock, BlockError> {
        let block = self.build();
        block.validate(max_threads, max_shared)?;
        Ok(block)
    }
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_block_1d() {
        let block = ThreadBlock::new_1d(256, 1024);
        assert_eq!(block.dim_x, 256);
        assert_eq!(block.dim_y, 1);
        assert_eq!(block.dim_z, 1);
        assert_eq!(block.total_threads(), 256);
        assert_eq!(block.shared_mem_per_block, 1024);
    }

    #[test]
    fn test_thread_block_2d() {
        let block = ThreadBlock::new_2d(16, 16, 2048);
        assert_eq!(block.total_threads(), 256);
    }

    #[test]
    fn test_thread_block_3d() {
        let block = ThreadBlock::new_3d(4, 8, 8, 4096);
        assert_eq!(block.total_threads(), 256);
    }

    #[test]
    fn test_block_covers_all_data() {
        // Ensure blocks can cover a data array completely
        let data_size = 10000u32;
        let threads_per_block = 256u32;
        let block = ThreadBlock::new_1d(threads_per_block, 0);
        let num_blocks = (data_size + threads_per_block - 1) / threads_per_block;
        let total_covered = num_blocks * threads_per_block;
        assert!(total_covered >= data_size);
        assert_eq!(num_blocks, 40); // ceil(10000/256)

        // Verify per-block ranges cover all indices
        let mut covered = vec![false; data_size as usize];
        for b in 0..num_blocks {
            let start = b * threads_per_block;
            let end = std::cmp::min(start + threads_per_block, data_size);
            for i in start..end {
                covered[i as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "All data elements should be covered");
    }

    #[test]
    fn test_warp_count_correct() {
        let block = ThreadBlock::new_1d(256, 0);
        let warp_alloc = WarpAllocation::from_block(&block, 32);
        assert_eq!(warp_alloc.num_warps, 8);
        assert_eq!(warp_alloc.full_warp_count(), 8);
        assert!(!warp_alloc.has_partial_warp());

        let block2 = ThreadBlock::new_1d(300, 0);
        let warp_alloc2 = WarpAllocation::from_block(&block2, 32);
        assert_eq!(warp_alloc2.num_warps, 10); // ceil(300/32) = 10
        assert_eq!(warp_alloc2.full_warp_count(), 9);
        assert!(warp_alloc2.has_partial_warp());
    }

    #[test]
    fn test_warp_ranges() {
        let block = ThreadBlock::new_1d(96, 0);
        let warp_alloc = WarpAllocation::from_block(&block, 32);
        assert_eq!(warp_alloc.num_warps, 3);

        assert_eq!(warp_alloc.warp_range(0), Some((0, 32)));
        assert_eq!(warp_alloc.warp_range(1), Some((32, 64)));
        assert_eq!(warp_alloc.warp_range(2), Some((64, 96)));
        assert_eq!(warp_alloc.warp_range(3), None);

        assert_eq!(warp_alloc.threads_for_warp(0), 32);
        assert_eq!(warp_alloc.threads_for_warp(2), 32);
    }

    #[test]
    fn test_warp_partial() {
        let block = ThreadBlock::new_1d(100, 0);
        let warp_alloc = WarpAllocation::from_block(&block, 32);
        assert_eq!(warp_alloc.num_warps, 4); // ceil(100/32)
        assert_eq!(warp_alloc.threads_for_warp(3), 4); // 100 - 96
        assert!(warp_alloc.has_partial_warp());
    }

    #[test]
    fn test_occupancy_calculation() {
        let block = ThreadBlock::new_1d(256, 0);
        let scheduler = BlockScheduler::new(10, 32, 2048);
        let occupancy = calculate_occupancy(&block, &scheduler, 32, 64, 49152);

        assert_eq!(occupancy.active_blocks_per_sm, 8); // min(2048/256=8, 64/8=8, 32, 49152/0→32)
        assert_eq!(occupancy.warps_per_sm, 64); // 8 blocks * 8 warps
        assert!((occupancy.occupancy_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_occupancy_with_shared_mem() {
        let block = ThreadBlock::new_1d(256, 8192);
        let scheduler = BlockScheduler::new(10, 32, 2048);
        let occupancy = calculate_occupancy(&block, &scheduler, 32, 64, 49152);

        // 49152 / 8192 = 6 blocks by shared mem
        assert_eq!(occupancy.active_blocks_per_sm, 6);
        assert_eq!(occupancy.warps_per_sm, 48); // 6 * 8
        assert!((occupancy.occupancy_ratio - (48.0 / 64.0)).abs() < 1e-9);
    }

    #[test]
    fn test_scheduler_assigns_all_blocks_round_robin() {
        let block = ThreadBlock::new_1d(128, 0);
        let scheduler = BlockScheduler::new(4, 16, 2048);
        let total_blocks = 20u32;
        let assignments = scheduler.schedule_round_robin(total_blocks, &block);

        assert_eq!(assignments.len(), 4);
        assert!(scheduler.all_blocks_scheduled(&assignments, total_blocks));

        // Round-robin should distribute roughly evenly
        let counts: Vec<u32> = assignments.iter().map(|a| a.block_count()).collect();
        let max = *counts.iter().max().unwrap();
        let min = *counts.iter().min().unwrap();
        assert!(max - min <= 1);
    }

    #[test]
    fn test_scheduler_assigns_all_blocks_greedy() {
        let block = ThreadBlock::new_1d(128, 0);
        let scheduler = BlockScheduler::new(4, 16, 2048);
        let total_blocks = 20u32;
        let assignments = scheduler.schedule_greedy(total_blocks, &block);

        assert!(scheduler.all_blocks_scheduled(&assignments, total_blocks));

        // Greedy fills SM 0 first
        assert_eq!(assignments[0].block_count(), 16); // max_blocks_per_sm
        assert_eq!(assignments[1].block_count(), 4);
        assert_eq!(assignments[2].block_count(), 0);
    }

    #[test]
    fn test_scheduler_respects_thread_limit() {
        let block = ThreadBlock::new_1d(256, 0);
        let scheduler = BlockScheduler::new(2, 32, 512); // Only 512 threads per SM
        let assignments = scheduler.schedule_greedy(100, &block);

        // 512 / 256 = 2 blocks per SM max
        assert_eq!(assignments[0].block_count(), 2);
        assert_eq!(assignments[1].block_count(), 2);
    }

    #[test]
    fn test_config_builder() {
        let block = BlockConfig::new()
            .dim_x(128)
            .dim_y(2)
            .dim_z(1)
            .shared_mem(4096)
            .build();

        assert_eq!(block.dim_x, 128);
        assert_eq!(block.dim_y, 2);
        assert_eq!(block.dim_z, 1);
        assert_eq!(block.total_threads(), 256);
        assert_eq!(block.shared_mem_per_block, 4096);
    }

    #[test]
    fn test_config_builder_validated_success() {
        let block = BlockConfig::new()
            .dim_x(256)
            .shared_mem(1024)
            .build_validated(1024, 49152);

        assert!(block.is_ok());
        let b = block.unwrap();
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn test_config_builder_validated_too_many_threads() {
        let result = BlockConfig::new()
            .dim_x(2048)
            .build_validated(1024, 49152);

        assert!(matches!(result, Err(BlockError::TooManyThreads { .. })));
    }

    #[test]
    fn test_config_builder_validated_too_much_shared() {
        let result = BlockConfig::new()
            .dim_x(64)
            .shared_mem(65536)
            .build_validated(1024, 49152);

        assert!(matches!(result, Err(BlockError::TooMuchSharedMem { .. })));
    }

    #[test]
    fn test_validate_zero_dimension() {
        let block = ThreadBlock::new_1d(0, 0);
        let result = block.validate(1024, 49152);
        assert!(matches!(result, Err(BlockError::ZeroDimension)));
    }

    #[test]
    fn test_blocks_per_sm_shared_mem() {
        let scheduler = BlockScheduler::new(10, 32, 2048);
        // 49152 total / 8192 per block = 6
        assert_eq!(scheduler.blocks_per_sm_shared_mem(8192, 49152), 6);
        // No shared mem used → max blocks
        assert_eq!(scheduler.blocks_per_sm_shared_mem(0, 49152), 32);
    }

    #[test]
    fn test_sm_assignment_total_threads() {
        let block = ThreadBlock::new_1d(128, 0);
        let assignment = SmAssignment {
            sm_id: 0,
            blocks: vec![0, 1, 2],
        };
        assert_eq!(assignment.block_count(), 3);
        assert_eq!(assignment.total_threads(&block), 384);
    }

    #[test]
    fn test_block_error_display() {
        let err = BlockError::TooManyThreads { requested: 2048, maximum: 1024 };
        assert_eq!(format!("{}", err), "requested 2048 threads but maximum is 1024");

        let err2 = BlockError::ZeroDimension;
        assert_eq!(format!("{}", err2), "block dimension cannot be zero");
    }
}
