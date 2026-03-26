# HW4: FlashAttention Forward Pass (CUDA & HIP)
This repository contains an optimized CUDA implementation of the FlashAttention forward pass, designed to overcome the memory-bound limitations of standard attention mechanisms.

## Implementation Overview
The core objective of this project is to eliminate the need to materialize the massive $N \times N$ score matrix in global memory (HBM) by using Kernel Fusion and Tiling.
### Core Algorithms
- Tiling & SRAM Management
- Partial Softmax
- Numerical Stability
### Configuration & Optimization
Based on hardware constraints and performance analysis, the following configurations were chosen:
- Tile Sizes:
  - $$B_r$$=64 (Query rows per block)
  - $$B_c$$=32 (Key/Value rows per block)
- Parallelism Strategy:
  - Each thread is responsible for an entire $$1\times d$$ query row.
- Performance Boosts:
  - Vectorization: Utilizing float4 for dot-product calculations achieved a 1.3x speedup over the basic GPU version.
  - Partial Softmax: Implementing the online softmax reduced execution time from 10.9s to 1.2s.
## Profiling Results (GTX 1080)
- SM Efficiency: 99.96%
- Achieved Occupancy: ~0.62
- Shared Memory Throughput: 599.46 GB/s
- Global Memory Load: 262.23 GB/s
