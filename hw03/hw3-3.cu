#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

// ======================
// Configuration & Types
// ======================

#define BLOCK_SIZE 64
#define HALF_BLOCK 32

const int INF = (1 << 30) - 1;

// Original number of vertices and edges
int V, E;
// Padded dimension (multiple of BLOCK_SIZE)
int n_padded;

// Host distance matrix (pinned)
int *Dist = nullptr;
// Device matrices (one per GPU)
int *Dist_GPU[2] = { nullptr, nullptr };

// ======================
// Utility: CUDA error check (optional but helpful)
// ======================
inline void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

// ======================
// I/O
// ======================

void input(const char *inFileName) {
    FILE *file = fopen(inFileName, "rb");
    if (!file) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }

    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Pad to multiple of BLOCK_SIZE
    int remainder = V % BLOCK_SIZE;
    if (remainder == 0)
        n_padded = V;
    else
        n_padded = V + (BLOCK_SIZE - remainder);

    // Allocate pinned host memory for Dist
    checkCuda(cudaMallocHost(&Dist, n_padded * n_padded * sizeof(int)), "cudaMallocHost(Dist)");

    // Initialize distance matrix
    for (int i = 0; i < n_padded; ++i) {
        for (int j = 0; j < n_padded; ++j) {
            if (i == j && i < V)
                Dist[i * n_padded + j] = 0;
            else
                Dist[i * n_padded + j] = INF;
        }
    }

    // Read edges
    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        int u = pair[0];
        int v = pair[1];
        int w = pair[2];
        Dist[u * n_padded + v] = w;
    }

    fclose(file);
}

void output(const char *outFileName) {
    FILE *file = fopen(outFileName, "wb");
    if (!file) {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }

    // Write only the original V x V portion
    for (int i = 0; i < V; ++i) {
        fwrite(&Dist[i * n_padded], sizeof(int), V, file);
    }

    fclose(file);

    if (Dist)
        cudaFreeHost(Dist);
}

// ======================
// CUDA kernels (3-phase blocked FW)
// ======================

// Phase 1: update the pivot block (r, r)
__global__ void phase1(int *D, int r, int n) {
    __shared__ int s[BLOCK_SIZE * BLOCK_SIZE];

    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31

    int base_i = (r * BLOCK_SIZE);
    int base_j = (r * BLOCK_SIZE);

    // Load 64x64 block (r,r) into shared memory using 32x32 threads
    s[ty * BLOCK_SIZE + tx] =
        D[(base_i + ty) * n + (base_j + tx)];
    s[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_i + ty) * n + (base_j + tx + HALF_BLOCK)];
    s[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx)];
    s[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx + HALF_BLOCK)];

    __syncthreads();

    // Floyd–Warshall within the pivot block
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        __syncthreads();

        int val_00 = s[ty * BLOCK_SIZE + tx];
        int val_01 = s[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
        int val_10 = s[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
        int val_11 = s[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];

        int s_yk_0  = s[ty * BLOCK_SIZE + k];
        int s_yk_1  = s[(ty + HALF_BLOCK) * BLOCK_SIZE + k];
        int s_kx_0  = s[k * BLOCK_SIZE + tx];
        int s_kx_1  = s[k * BLOCK_SIZE + (tx + HALF_BLOCK)];

        val_00 = min(val_00, s_yk_0 + s_kx_0);
        val_01 = min(val_01, s_yk_0 + s_kx_1);
        val_10 = min(val_10, s_yk_1 + s_kx_0);
        val_11 = min(val_11, s_yk_1 + s_kx_1);

        s[ty * BLOCK_SIZE + tx] = val_00;
        s[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] = val_01;
        s[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] = val_10;
        s[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] = val_11;
    }

    __syncthreads();

    // Store back pivot block to global memory
    D[(base_i + ty) * n + (base_j + tx)] =
        s[ty * BLOCK_SIZE + tx];
    D[(base_i + ty) * n + (base_j + tx + HALF_BLOCK)] =
        s[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
    D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx)] =
        s[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
    D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx + HALF_BLOCK)] =
        s[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];
}

// Phase 2: update pivot row and pivot column (blocks (r, j) and (i, r), j≠r, i≠r)
// This is a slightly cleaned-up version of the second code's Phase 2 (for clarity),
// but runs on each GPU independently (they both hold full Dist).
__global__ void phase2(int *D, int r, int n) {
    int blockID = blockIdx.y;  // which block along row/col we’re updating

    if (blockID == r) return;  // skip pivot block itself

    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31

    // Shared memory tiles
    __shared__ int pivotBlock[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int pivotCol[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ int pivotRow[BLOCK_SIZE * BLOCK_SIZE];

    int base_pivot_i = r * BLOCK_SIZE;
    int base_pivot_j = r * BLOCK_SIZE;

    int base_col_i = blockID * BLOCK_SIZE;
    int base_col_j = r * BLOCK_SIZE;

    int base_row_i = r * BLOCK_SIZE;
    int base_row_j = blockID * BLOCK_SIZE;

    // Load pivot block
    pivotBlock[ty * BLOCK_SIZE + tx] =
        D[(base_pivot_i + ty) * n + (base_pivot_j + tx)];
    pivotBlock[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_pivot_i + ty) * n + (base_pivot_j + tx + HALF_BLOCK)];
    pivotBlock[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_pivot_i + ty + HALF_BLOCK) * n + (base_pivot_j + tx)];
    pivotBlock[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_pivot_i + ty + HALF_BLOCK) * n + (base_pivot_j + tx + HALF_BLOCK)];

    // Load pivot column block (blockID, r)
    pivotCol[ty * BLOCK_SIZE + tx] =
        D[(base_col_i + ty) * n + (base_col_j + tx)];
    pivotCol[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_col_i + ty) * n + (base_col_j + tx + HALF_BLOCK)];
    pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_col_i + ty + HALF_BLOCK) * n + (base_col_j + tx)];
    pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_col_i + ty + HALF_BLOCK) * n + (base_col_j + tx + HALF_BLOCK)];

    // Load pivot row block (r, blockID)
    pivotRow[ty * BLOCK_SIZE + tx] =
        D[(base_row_i + ty) * n + (base_row_j + tx)];
    pivotRow[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_row_i + ty) * n + (base_row_j + tx + HALF_BLOCK)];
    pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_row_i + ty + HALF_BLOCK) * n + (base_row_j + tx)];
    pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_row_i + ty + HALF_BLOCK) * n + (base_row_j + tx + HALF_BLOCK)];

    __syncthreads();

    // Update both pivot column and pivot row blocks
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        // pivot column: (blockID, r)
        int col_00 = pivotCol[ty * BLOCK_SIZE + tx];
        int col_01 = pivotCol[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
        int col_10 = pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
        int col_11 = pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];

        int col_yk_0  = pivotCol[ty * BLOCK_SIZE + k];
        int col_yk_1  = pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + k];
        int piv_kx_0  = pivotBlock[k * BLOCK_SIZE + tx];
        int piv_kx_1  = pivotBlock[k * BLOCK_SIZE + (tx + HALF_BLOCK)];

        col_00 = min(col_00, col_yk_0 + piv_kx_0);
        col_01 = min(col_01, col_yk_0 + piv_kx_1);
        col_10 = min(col_10, col_yk_1 + piv_kx_0);
        col_11 = min(col_11, col_yk_1 + piv_kx_1);

        pivotCol[ty * BLOCK_SIZE + tx] = col_00;
        pivotCol[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] = col_01;
        pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] = col_10;
        pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] = col_11;

        // pivot row: (r, blockID)
        int row_00 = pivotRow[ty * BLOCK_SIZE + tx];
        int row_01 = pivotRow[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
        int row_10 = pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
        int row_11 = pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];

        int piv_yk_0  = pivotBlock[ty * BLOCK_SIZE + k];
        int piv_yk_1  = pivotBlock[(ty + HALF_BLOCK) * BLOCK_SIZE + k];
        int row_kx_0  = pivotRow[k * BLOCK_SIZE + tx];
        int row_kx_1  = pivotRow[k * BLOCK_SIZE + (tx + HALF_BLOCK)];

        row_00 = min(row_00, piv_yk_0 + row_kx_0);
        row_01 = min(row_01, piv_yk_0 + row_kx_1);
        row_10 = min(row_10, piv_yk_1 + row_kx_0);
        row_11 = min(row_11, piv_yk_1 + row_kx_1);

        pivotRow[ty * BLOCK_SIZE + tx] = row_00;
        pivotRow[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] = row_01;
        pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] = row_10;
        pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] = row_11;
    }

    __syncthreads();

    // Store updated pivot column block
    D[(base_col_i + ty) * n + (base_col_j + tx)] =
        pivotCol[ty * BLOCK_SIZE + tx];
    D[(base_col_i + ty) * n + (base_col_j + tx + HALF_BLOCK)] =
        pivotCol[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
    D[(base_col_i + ty + HALF_BLOCK) * n + (base_col_j + tx)] =
        pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
    D[(base_col_i + ty + HALF_BLOCK) * n + (base_col_j + tx + HALF_BLOCK)] =
        pivotCol[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];

    // Store updated pivot row block
    D[(base_row_i + ty) * n + (base_row_j + tx)] =
        pivotRow[ty * BLOCK_SIZE + tx];
    D[(base_row_i + ty) * n + (base_row_j + tx + HALF_BLOCK)] =
        pivotRow[ty * BLOCK_SIZE + (tx + HALF_BLOCK)];
    D[(base_row_i + ty + HALF_BLOCK) * n + (base_row_j + tx)] =
        pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + tx];
    D[(base_row_i + ty + HALF_BLOCK) * n + (base_row_j + tx + HALF_BLOCK)] =
        pivotRow[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)];
}

// Phase 3: update all remaining blocks (i, j) where i != r and j != r
// Each GPU only updates the row-blocks it owns (based on 'startBlock' and 'numBlocks').
__global__ void phase3(int *D, int r, int n, int startBlock) {
    __shared__ int s[2 * BLOCK_SIZE * BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // blockIdx.x: index over row-blocks assigned to this GPU (0..numBlocks-1)
    // blockIdx.y: index over all column-blocks except pivot (0..numRounds-2)
    int block_i = startBlock + blockIdx.x;
    int block_j = blockIdx.y;
    if (block_j >= r) block_j++;   // skip pivot column

    int base_i = block_i * BLOCK_SIZE;
    int base_j = block_j * BLOCK_SIZE;
    int base_k = r * BLOCK_SIZE;

    // Load current C block (i,j) from global memory
    int val_00 = D[(base_i + ty) * n + (base_j + tx)];
    int val_01 = D[(base_i + ty) * n + (base_j + tx + HALF_BLOCK)];
    int val_10 = D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx)];
    int val_11 = D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx + HALF_BLOCK)];

    // s[0 .. BLOCK_SIZE*BLOCK_SIZE-1]  : A block (i, r)
    // s[BLOCK_SIZE*BLOCK_SIZE .. ]     : B block (r, j)
    int *A = s;
    int *B = s + BLOCK_SIZE * BLOCK_SIZE;

    // Load A = block(i, r)
    A[ty * BLOCK_SIZE + tx] =
        D[(base_i + ty) * n + (base_k + tx)];
    A[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_i + ty) * n + (base_k + tx + HALF_BLOCK)];
    A[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_i + ty + HALF_BLOCK) * n + (base_k + tx)];
    A[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_i + ty + HALF_BLOCK) * n + (base_k + tx + HALF_BLOCK)];

    // Load B = block(r, j)
    B[ty * BLOCK_SIZE + tx] =
        D[(base_k + ty) * n + (base_j + tx)];
    B[ty * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_k + ty) * n + (base_j + tx + HALF_BLOCK)];
    B[(ty + HALF_BLOCK) * BLOCK_SIZE + tx] =
        D[(base_k + ty + HALF_BLOCK) * n + (base_j + tx)];
    B[(ty + HALF_BLOCK) * BLOCK_SIZE + (tx + HALF_BLOCK)] =
        D[(base_k + ty + HALF_BLOCK) * n + (base_j + tx + HALF_BLOCK)];

    __syncthreads();

    // Update C(i,j) = min( C(i,j), A(i,k) + B(k,j) )
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int a_00 = A[ty * BLOCK_SIZE + k];
        int a_10 = A[(ty + HALF_BLOCK) * BLOCK_SIZE + k];

        int b_00 = B[k * BLOCK_SIZE + tx];
        int b_01 = B[k * BLOCK_SIZE + (tx + HALF_BLOCK)];

        val_00 = min(val_00, a_00 + b_00);
        val_01 = min(val_01, a_00 + b_01);
        val_10 = min(val_10, a_10 + b_00);
        val_11 = min(val_11, a_10 + b_01);
    }

    // Store back updated block (i,j)
    D[(base_i + ty) * n + (base_j + tx)] =
        val_00;
    D[(base_i + ty) * n + (base_j + tx + HALF_BLOCK)] =
        val_01;
    D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx)] =
        val_10;
    D[(base_i + ty + HALF_BLOCK) * n + (base_j + tx + HALF_BLOCK)] =
        val_11;
}

// ======================
// Blocked Floyd–Warshall driver (multi-GPU, 2 devices)
// ======================

void blocked_FW_multiGPU() {
    int numRounds = n_padded / BLOCK_SIZE;

#pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();   // 0 or 1
        int dev = tid;                    // device 0 and 1

        // Partition rows between the two GPUs in block units
        int startBlock = (numRounds / 2) * tid;
        int numBlocks  = (numRounds / 2) + (numRounds % 2) * (tid == 1);

        checkCuda(cudaSetDevice(dev), "cudaSetDevice");

        // Allocate device memory
        checkCuda(cudaMalloc(&Dist_GPU[tid],
                             n_padded * n_padded * sizeof(int)),
                  "cudaMalloc(Dist_GPU)");

#pragma omp barrier
        // Copy owned rows from host to device
        checkCuda(cudaMemcpy(Dist_GPU[tid] + startBlock * BLOCK_SIZE * n_padded,
                             Dist + startBlock * BLOCK_SIZE * n_padded,
                             (size_t)numBlocks * BLOCK_SIZE * n_padded * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D local slice");

        dim3 threads(32, 32);

        for (int r = 0; r < numRounds; ++r) {
            // Peer sync of pivot row: whichever GPU owns the pivot row sends it to the other
            int ownsPivotRow = (r >= startBlock && r < startBlock + numBlocks);

            if (ownsPivotRow) {
                // Send pivot row to the other device
                int other = 1 - tid;
                size_t rowOffset = (size_t)r * BLOCK_SIZE * n_padded;
                checkCuda(cudaMemcpyPeer(Dist_GPU[other] + rowOffset, other,
                                         Dist_GPU[tid] + rowOffset, dev,
                                         BLOCK_SIZE * n_padded * sizeof(int)),
                          "cudaMemcpyPeer pivot row");
            }

#pragma omp barrier
            // Both devices now have the updated pivot row
            // ---- Phase 1 ----
            {
                dim3 gridPhase1(1, 1);
                phase1<<<gridPhase1, threads>>>(Dist_GPU[tid], r, n_padded);
                checkCuda(cudaDeviceSynchronize(), "phase1 sync");
            }

            // ---- Phase 2 ---- (pivot row & column; each device does full work)
            {
                dim3 gridPhase2(1, numRounds);
                phase2<<<gridPhase2, threads>>>(Dist_GPU[tid], r, n_padded);
                checkCuda(cudaDeviceSynchronize(), "phase2 sync");
            }

            // ---- Phase 3 ---- (only blocks in this device's row range)
            {
                int colBlocks = numRounds - 1; // excluding pivot column
                dim3 gridPhase3(numBlocks, colBlocks);
                phase3<<<gridPhase3, threads>>>(Dist_GPU[tid], r, n_padded, startBlock);
                checkCuda(cudaDeviceSynchronize(), "phase3 sync");
            }
        }

        // Copy back this device's rows
        checkCuda(cudaMemcpy(Dist + startBlock * BLOCK_SIZE * n_padded,
                             Dist_GPU[tid] + startBlock * BLOCK_SIZE * n_padded,
                             (size_t)numBlocks * BLOCK_SIZE * n_padded * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H local slice");

#pragma omp barrier

        // Free device memory
        cudaFree(Dist_GPU[tid]);
    }
}

// ======================
// main
// ======================

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Read graph
    input(argv[1]);

    // Run blocked Floyd–Warshall on two GPUs
    blocked_FW_multiGPU();

    // Write result
    output(argv[2]);

    return 0;
}
