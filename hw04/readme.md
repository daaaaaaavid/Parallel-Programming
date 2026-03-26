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
-- $$B_r$$
