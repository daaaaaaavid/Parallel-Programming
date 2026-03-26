#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define DEV_NO 0
cudaDeviceProp prop;
const int INF = 1073741823;  // (1<<30)-1

// Forward declarations
void input(const char *inFileName);
void output(const char *outFileName);
void block_FW();

__global__ void phase1(int r, int *Dist_GPU, int n);
__global__ void phase2(int r, int *Dist_GPU, int n);
__global__ void phase3(int r, int *Dist_GPU, int n);

// Globals
int n, m;          // original n, number of edges
int n_pad;         // padded n (multiple of 64)
int *Dist;         // host matrix (n_pad x n_pad, flat)
int *Dist_GPU;     // device matrix

// ==================== main ====================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return 1;
    }

    input(argv[1]);
    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    block_FW();
    output(argv[2]);

    return 0;
}

// ==================== I/O ====================

void input(const char *inFileName) {
    FILE *file = fopen(inFileName, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open input file\n");
        exit(1);
    }

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // pad n up to multiple of 64
    const int B = 64;
    n_pad = ( (n + B - 1) / B ) * B;

    // allocate pinned host memory for better H2D/D2H bandwidth
    if (cudaMallocHost(&Dist, (size_t)n_pad * n_pad * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost(Dist) failed\n");
        exit(1);
    }

    // initialize padded matrix
    // top-left n x n: 0 on diag, INF elsewhere
    // rest of padded region: INF
    for (int i = 0; i < n_pad; ++i) {
        for (int j = 0; j < n_pad; ++j) {
            if (i < n && j < n) {
                Dist[i * n_pad + j] = (i == j ? 0 : INF);
            } else {
                Dist[i * n_pad + j] = INF;
            }
        }
    }

    // read edges
    int edge[3];
    for (int e = 0; e < m; ++e) {
        fread(edge, sizeof(int), 3, file);
        int u = edge[0];
        int v = edge[1];
        int w = edge[2];
        // assume 0 <= u,v < n
        Dist[u * n_pad + v] = w;
    }

    fclose(file);
}

void output(const char *outFileName) {
    FILE *file = fopen(outFileName, "wb");
    if (!file) {
        fprintf(stderr, "Cannot open output file\n");
        exit(1);
    }

    // write only top-left n x n region
    for (int i = 0; i < n; ++i) {
        fwrite(&Dist[i * n_pad], sizeof(int), n, file);
    }

    fclose(file);
    cudaFreeHost(Dist);
}

// ==================== Blocked Floyd–Warshall ====================

void block_FW() {
    size_t bytes = (size_t)n_pad * n_pad * sizeof(int);

    // allocate device memory
    if (cudaMalloc(&Dist_GPU, bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(Dist_GPU) failed\n");
        exit(1);
    }

    // copy host → device
    cudaMemcpy(Dist_GPU, Dist, bytes, cudaMemcpyHostToDevice);

    const int B = 64;
    int rounds = n_pad / B;   // number of 64x64 tiles along one dimension

    dim3 block(32, 32);

    // Phase 2: grid(2, rounds-1)  (one row pass, one col pass)
    // Phase 3: grid(rounds-1, rounds-1)
    for (int r = 0; r < rounds; ++r) {
        // phase 1: pivot block (r,r)
        phase1<<<1, block>>>(r, Dist_GPU, n_pad);

        // phase 2: tiles in pivot row & pivot column
        phase2<<<dim3(2, rounds - 1), block>>>(r, Dist_GPU, n_pad);

        // phase 3: remaining tiles
        phase3<<<dim3(rounds - 1, rounds - 1), block>>>(r, Dist_GPU, n_pad);

        cudaDeviceSynchronize(); // one sync per round
    }

    // copy back full padded matrix
    cudaMemcpy(Dist, Dist_GPU, bytes, cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU);
}

// ==================== Kernels ====================
//
// Each block is 32x32 threads, but the tile is 64x64.
// So each thread is responsible for 4 elements:
//   (i, j), (i, j+32), (i+32, j), (i+32, j+32)
//

// ---- Phase 1: update pivot tile (r, r) ----

__global__ void phase1(int r, int *Dist_GPU, int n) {
    __shared__ int s[64 * 64];

    int i = threadIdx.y;  // 0..31
    int j = threadIdx.x;  // 0..31

    int base = r << 6;    // r * 64

    int bi = base + i;
    int bj = base + j;

    // load 4 values into shared memory (forming the 64x64 tile)
    s[i * 64 + j]                   = Dist_GPU[(bi      ) * n + (bj      )];
    s[i * 64 + (j + 32)]            = Dist_GPU[(bi      ) * n + (bj + 32)];
    s[(i + 32) * 64 + j]            = Dist_GPU[(bi + 32 ) * n + (bj      )];
    s[(i + 32) * 64 + (j + 32)]     = Dist_GPU[(bi + 32 ) * n + (bj + 32)];

    // Floyd–Warshall inside the pivot tile
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        int s_ik     = s[i        * 64 + k];
        int s_i32k   = s[(i + 32) * 64 + k];

        // upper-left
        int via0 = s_ik + s[k * 64 + j];
        if (via0 < s[i * 64 + j]) s[i * 64 + j] = via0;

        // upper-right
        int via1 = s_ik + s[k * 64 + (j + 32)];
        if (via1 < s[i * 64 + (j + 32)]) s[i * 64 + (j + 32)] = via1;

        // lower-left
        int via2 = s_i32k + s[k * 64 + j];
        if (via2 < s[(i + 32) * 64 + j]) s[(i + 32) * 64 + j] = via2;

        // lower-right
        int via3 = s_i32k + s[k * 64 + (j + 32)];
        if (via3 < s[(i + 32) * 64 + (j + 32)]) s[(i + 32) * 64 + (j + 32)] = via3;
    }

    // write back to global memory
    Dist_GPU[(bi      ) * n + (bj      )] = s[i * 64 + j];
    Dist_GPU[(bi      ) * n + (bj + 32)]  = s[i * 64 + (j + 32)];
    Dist_GPU[(bi + 32 ) * n + (bj      )] = s[(i + 32) * 64 + j];
    Dist_GPU[(bi + 32 ) * n + (bj + 32)]  = s[(i + 32) * 64 + (j + 32)];
}

// ---- Phase 2: update row & column tiles (using fused kernel) ----
//
// gridDim.x = 2, gridDim.y = rounds-1
//   blockIdx.x = 0 → column tiles (b, r)
//   blockIdx.x = 1 → row   tiles (r, b)
//

__global__ void phase2(int r, int *Dist_GPU, int n) {
    __shared__ int s[2 * 64 * 64];  // first 64x64 for (i,k), second 64x64 for (k,j)

    int i = threadIdx.y;  // 0..31
    int j = threadIdx.x;  // 0..31

    int rounds = gridDim.y + 1;   // because we've launched with (2, rounds-1)
    (void)rounds;                 // silence unused warning if any

    // Determine which tile this block is processing
    // ROW: blockIdx.x = 1 -> (r, b != r)
    // COL: blockIdx.x = 0 -> (b != r, r)
    int b_i, b_j;
    if (blockIdx.x == 1) {
        // row tiles (r, b)
        int b = blockIdx.y;
        if (b >= r) b += 1;      // skip r
        b_i = r;
        b_j = b;
    } else {
        // column tiles (b, r)
        int b = blockIdx.y;
        if (b >= r) b += 1;      // skip r
        b_i = b;
        b_j = r;
    }

    int base_i = b_i << 6;   // b_i * 64
    int base_j = b_j << 6;   // b_j * 64
    int base_k = r   << 6;   // r   * 64

    int gi0 = base_i + i;
    int gj0 = base_j + j;

    // load original tile values into registers (4 cells per thread)
    int val0 = Dist_GPU[(gi0      ) * n + (gj0      )];
    int val1 = Dist_GPU[(gi0      ) * n + (gj0 + 32)];
    int val2 = Dist_GPU[(gi0 + 32 ) * n + (gj0      )];
    int val3 = Dist_GPU[(gi0 + 32 ) * n + (gj0 + 32)];

    // shared memory:
    // s[0..4095]     : D(b_i, k)  (rows)
    // s[4096..8191]  : D(k, b_j)  (cols)

    // load D(b_i, k) into s
    s[i * 64 + j]                   = Dist_GPU[(base_i + i      ) * n + (base_k + j      )];
    s[i * 64 + (j + 32)]            = Dist_GPU[(base_i + i      ) * n + (base_k + j + 32)];
    s[(i + 32) * 64 + j]            = Dist_GPU[(base_i + i + 32 ) * n + (base_k + j      )];
    s[(i + 32) * 64 + (j + 32)]     = Dist_GPU[(base_i + i + 32 ) * n + (base_k + j + 32)];

    // load D(k, b_j) into s + 4096
    int offset = 64 * 64;  // 4096
    s[offset + i * 64 + j]                   = Dist_GPU[(base_k + i      ) * n + (base_j + j      )];
    s[offset + i * 64 + (j + 32)]            = Dist_GPU[(base_k + i      ) * n + (base_j + j + 32)];
    s[offset + (i + 32) * 64 + j]            = Dist_GPU[(base_k + i + 32 ) * n + (base_j + j      )];
    s[offset + (i + 32) * 64 + (j + 32)]     = Dist_GPU[(base_k + i + 32 ) * n + (base_j + j + 32)];

    __syncthreads();

    // FW update across k in the pivot tile
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int left0  = s[i        * 64 + k];
        int left32 = s[(i + 32) * 64 + k];

        int right0  = s[offset + k * 64 + j];
        int right32 = s[offset + k * 64 + (j + 32)];

        int via0 = left0 + right0;
        if (via0 < val0) val0 = via0;

        int via1 = left0 + right32;
        if (via1 < val1) val1 = via1;

        int via2 = left32 + right0;
        if (via2 < val2) val2 = via2;

        int via3 = left32 + right32;
        if (via3 < val3) val3 = via3;
    }

    // write back updated tile
    Dist_GPU[(gi0      ) * n + (gj0      )] = val0;
    Dist_GPU[(gi0      ) * n + (gj0 + 32)]  = val1;
    Dist_GPU[(gi0 + 32 ) * n + (gj0      )] = val2;
    Dist_GPU[(gi0 + 32 ) * n + (gj0 + 32)]  = val3;
}

// ---- Phase 3: update all remaining tiles (by, bx), where by != r, bx != r ----

__global__ void phase3(int r, int *Dist_GPU, int n) {
    __shared__ int s[2 * 64 * 64];  // s for (i,k) and (k,j)

    int i = threadIdx.y;  // 0..31
    int j = threadIdx.x;  // 0..31

    // Skip row/col r by shifting indices
    int by = blockIdx.x;
    if (by >= r) by += 1;

    int bx = blockIdx.y;
    if (bx >= r) bx += 1;

    int base_i = by << 6;  // by * 64
    int base_j = bx << 6;  // bx * 64
    int base_k = r  << 6;  // r  * 64

    int gi0 = base_i + i;
    int gj0 = base_j + j;

    // load original values for this tile into registers
    int val0 = Dist_GPU[(gi0      ) * n + (gj0      )];
    int val1 = Dist_GPU[(gi0      ) * n + (gj0 + 32)];
    int val2 = Dist_GPU[(gi0 + 32 ) * n + (gj0      )];
    int val3 = Dist_GPU[(gi0 + 32 ) * n + (gj0 + 32)];

    // shared memory layout same as phase2:
    // s[0..4095]     = D(by, r)  = rows
    // s[4096..8191]  = D(r, bx)  = cols

    // load D(by, r) → s
    s[i * 64 + j]                   = Dist_GPU[(base_i + i      ) * n + (base_k + j      )];
    s[i * 64 + (j + 32)]            = Dist_GPU[(base_i + i      ) * n + (base_k + j + 32)];
    s[(i + 32) * 64 + j]            = Dist_GPU[(base_i + i + 32 ) * n + (base_k + j      )];
    s[(i + 32) * 64 + (j + 32)]     = Dist_GPU[(base_i + i + 32 ) * n + (base_k + j + 32)];

    // load D(r, bx) → s + offset
    int offset = 64 * 64;
    s[offset + i * 64 + j]                   = Dist_GPU[(base_k + i      ) * n + (base_j + j      )];
    s[offset + i * 64 + (j + 32)]            = Dist_GPU[(base_k + i      ) * n + (base_j + j + 32)];
    s[offset + (i + 32) * 64 + j]            = Dist_GPU[(base_k + i + 32 ) * n + (base_j + j      )];
    s[offset + (i + 32) * 64 + (j + 32)]     = Dist_GPU[(base_k + i + 32 ) * n + (base_j + j + 32)];

    __syncthreads();

    // FW update
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int left0  = s[i        * 64 + k];
        int left32 = s[(i + 32) * 64 + k];

        int right0  = s[offset + k * 64 + j];
        int right32 = s[offset + k * 64 + (j + 32)];

        int via0 = left0 + right0;
        if (via0 < val0) val0 = via0;

        int via1 = left0 + right32;
        if (via1 < val1) val1 = via1;

        int via2 = left32 + right0;
        if (via2 < val2) val2 = via2;

        int via3 = left32 + right32;
        if (via3 < val3) val3 = via3;
    }

    // write back
    Dist_GPU[(gi0      ) * n + (gj0      )] = val0;
    Dist_GPU[(gi0      ) * n + (gj0 + 32)]  = val1;
    Dist_GPU[(gi0 + 32 ) * n + (gj0      )] = val2;
    Dist_GPU[(gi0 + 32 ) * n + (gj0 + 32)]  = val3;
}
