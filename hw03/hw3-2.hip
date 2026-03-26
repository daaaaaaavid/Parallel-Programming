// blocked_fw_hip.cu
// Cleaned HIP version of blocked (tiled) Floyd–Warshall.
// Keeps tile size = 64, each CUDA block (threads) handles 4 elements (32x32 threads).
// Build: hipcc -O3 -ffast-math -std=c++14 blocked_fw_hip.cu -o blocked_fw_hip

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) do {                                       \
    hipError_t e = cmd;                                           \
    if (e != hipSuccess) {                                        \
        fprintf(stderr, "HIP error %s:%d: '%s' (code %d)\n",      \
                __FILE__, __LINE__, hipGetErrorString(e), (int)e);\
        exit(EXIT_FAILURE);                                       \
    }                                                              \
} while(0)

constexpr int TILE = 64;          // tile dimension (fixed)
constexpr int BLOCK_DIM = 32;     // blockDim.x, blockDim.y (32x32 threads -> 1024 threads)
constexpr int INF = 1073741823;   // (1<<30)-1, large sentinel

// Forward declarations
void read_input(const char *inFile);
void write_output(const char *outFile);
void blocked_fw();

__global__ void phase1_kernel(int round, int *dDist, int n_pad);
__global__ void phase2_kernel(int round, int *dDist, int n_pad);
__global__ void phase3_kernel(int round, int *dDist, int n_pad);

// globals (kept simple)
int n;           // original n (vertices)
int m;           // number of edges
int n_pad;       // padded n (multiple of TILE)
int *hDist = nullptr;   // pinned host matrix (n_pad x n_pad)
int *dDist = nullptr;   // device matrix

// -------------------- main --------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(argv[1]);
    blocked_fw();
    write_output(argv[2]);

    return EXIT_SUCCESS;
}

// -------------------- I/O --------------------
void read_input(const char *inFile) {
    FILE *f = fopen(inFile, "rb");
    if (!f) {
        perror("fopen input");
        exit(EXIT_FAILURE);
    }

    // read n, m
    if (fread(&n, sizeof(int), 1, f) != 1) { perror("read n"); exit(EXIT_FAILURE); }
    if (fread(&m, sizeof(int), 1, f) != 1) { perror("read m"); exit(EXIT_FAILURE); }

    // pad to multiple of TILE
    n_pad = ((n + TILE - 1) / TILE) * TILE;
    size_t bytes = (size_t)n_pad * n_pad * sizeof(int);

    // pinned host allocation for faster H2D/D2H
    HIP_CHECK( hipHostMalloc((void**)&hDist, bytes, hipHostMallocDefault) );

    // initialize (padded)
    for (int i = 0; i < n_pad; ++i) {
        for (int j = 0; j < n_pad; ++j) {
            if (i < n && j < n) {
                hDist[i * n_pad + j] = (i == j ? 0 : INF);
            } else {
                hDist[i * n_pad + j] = INF;
            }
        }
    }

    // read edges (u, v, w) triplets
    for (int e = 0; e < m; ++e) {
        int edge[3];
        if (fread(edge, sizeof(int), 3, f) != 3) {
            fprintf(stderr, "error reading edge %d\n", e);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        int u = edge[0], v = edge[1], w = edge[2];
        // assume 0 <= u,v < n
        hDist[u * n_pad + v] = w;
    }

    fclose(f);
}

void write_output(const char *outFile) {
    FILE *f = fopen(outFile, "wb");
    if (!f) { perror("fopen output"); exit(EXIT_FAILURE); }

    // write top-left n x n
    for (int i = 0; i < n; ++i) {
        if (fwrite(&hDist[i * n_pad], sizeof(int), n, f) != (size_t)n) {
            perror("fwrite");
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
    HIP_CHECK( hipHostFree(hDist) );
    hDist = nullptr;
}

// -------------------- Blocked FW driver --------------------
void blocked_fw() {
    size_t bytes = (size_t)n_pad * n_pad * sizeof(int);

    // allocate device memory
    HIP_CHECK( hipMalloc((void**)&dDist, bytes) );

    // copy host -> device
    HIP_CHECK( hipMemcpy(dDist, hDist, bytes, hipMemcpyHostToDevice) );

    const int rounds = n_pad / TILE;

    // block config -- keep 32x32 threads to match 64x64 tile (4 elements/thread)
    dim3 block(BLOCK_DIM, BLOCK_DIM);

    // Launch phases for each round
    for (int r = 0; r < rounds; ++r) {
        // phase 1: pivot tile
        hipLaunchKernelGGL(phase1_kernel, dim3(1), block, 0, 0, r, dDist, n_pad);

        // phase 2: fused row & column (grid.x=2, grid.y=rounds-1)
        if (rounds > 1) {
            hipLaunchKernelGGL(phase2_kernel, dim3(2, rounds - 1), block, 0, 0, r, dDist, n_pad);
        }

        // phase 3: remaining tiles (rounds-1 x rounds-1)
        if (rounds > 1) {
            hipLaunchKernelGGL(phase3_kernel, dim3(rounds - 1, rounds - 1), block, 0, 0, r, dDist, n_pad);
        }

        // single sync per round
        HIP_CHECK( hipDeviceSynchronize() );
    }

    // copy device -> host
    HIP_CHECK( hipMemcpy(hDist, dDist, bytes, hipMemcpyDeviceToHost) );

    HIP_CHECK( hipFree(dDist) );
    dDist = nullptr;
}

// -------------------- Kernels --------------------
// Each kernel uses shared memory and implements blocked FW on 64x64 tiles.
// threadIdx: i (0..31), j (0..31)
// each thread handles four positions forming a 2x2 sub-block in the 64x64 tile.

__global__ void phase1_kernel(int r, int *dDist, int n) {
    extern __shared__ int s_mem[]; // not used because we declare static sized arrays below; keeping extern for portability

    __shared__ int s[TILE * TILE]; // 4096 ints

    int tidx = threadIdx.x; // 0..31
    int tidy = threadIdx.y; // 0..31

    int base = r * TILE;
    int bi = base + tidy;     // global row index for this thread's top half
    int bj = base + tidx;     // global col index for this thread's left half

    // load 4 elements per thread to form 64x64 tile in shared memory
    s[ tidy * TILE +  tidx ]                 = dDist[(bi      ) * n + (bj      )];
    s[ tidy * TILE + (tidx + 32) ]           = dDist[(bi      ) * n + (bj + 32 )];
    s[(tidy + 32) * TILE +  tidx ]           = dDist[(bi + 32 ) * n + (bj      )];
    s[(tidy + 32) * TILE + (tidx + 32) ]     = dDist[(bi + 32 ) * n + (bj + 32 )];

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        __syncthreads();
        int s_ik   = s[tidy * TILE + k];
        int s_i32k = s[(tidy + 32) * TILE + k];

        // compute via using k-th row/col inside tile
        int via0 = s_ik + s[k * TILE + tidx];
        if (via0 < s[tidy * TILE + tidx]) s[tidy * TILE + tidx] = via0;

        int via1 = s_ik + s[k * TILE + (tidx + 32)];
        if (via1 < s[ tidy * TILE + (tidx + 32) ]) s[ tidy * TILE + (tidx + 32) ] = via1;

        int via2 = s_i32k + s[k * TILE + tidx];
        if (via2 < s[(tidy + 32) * TILE + tidx]) s[(tidy + 32) * TILE + tidx] = via2;

        int via3 = s_i32k + s[k * TILE + (tidx + 32)];
        if (via3 < s[(tidy + 32) * TILE + (tidx + 32)]) s[(tidy + 32) * TILE + (tidx + 32)] = via3;
    }

    // write back
    dDist[(bi      ) * n + (bj      )] = s[ tidy * TILE + tidx ];
    dDist[(bi      ) * n + (bj + 32 )] = s[ tidy * TILE + (tidx + 32) ];
    dDist[(bi + 32 ) * n + (bj      )] = s[(tidy + 32) * TILE + tidx ];
    dDist[(bi + 32 ) * n + (bj + 32 )] = s[(tidy + 32) * TILE + (tidx + 32)];
}

__global__ void phase2_kernel(int r, int *dDist, int n) {
    __shared__ int s[2 * TILE * TILE]; // 8192 ints

    int tidx = threadIdx.x; // 0..31
    int tidy = threadIdx.y; // 0..31

    int gridY = gridDim.y; // rounds-1
    // decode which tile: blockIdx.x==1 -> row tile (r, b); blockIdx.x==0 -> col tile (b, r)
    int b;
    if (blockIdx.x == 1) {
        // row tile (r, b)
        b = blockIdx.y;
        if (b >= r) ++b;
        int base_i = r * TILE;
        int base_j = b * TILE;
        int base_k = r * TILE;

        int gi0 = base_i + tidy;
        int gj0 = base_j + tidx;

        // load the tile D(r, b) values to registers
        int val0 = dDist[(gi0      ) * n + (gj0      )];
        int val1 = dDist[(gi0      ) * n + (gj0 + 32 )];
        int val2 = dDist[(gi0 + 32 ) * n + (gj0      )];
        int val3 = dDist[(gi0 + 32 ) * n + (gj0 + 32 )];

        // load D(r, k) and D(k, b) to shared memory:
        // here base_i=r, base_k=r? For row case we need D(r,k) as rows and D(k,b) as cols
        // load D(r, k) into s[offset..)
        int base_row = base_i; // r
        int base_col = base_k; // r (used accordingly)
        int offset = TILE * TILE;

        // D(r,k)
        s[ tidy * TILE +  tidx ]                 = dDist[(base_row + tidy) * n + (base_k + tidx)];
        s[ tidy * TILE + (tidx + 32) ]           = dDist[(base_row + tidy) * n + (base_k + tidx + 32)];
        s[(tidy + 32) * TILE +  tidx ]           = dDist[(base_row + tidy + 32) * n + (base_k + tidx)];
        s[(tidy + 32) * TILE + (tidx + 32) ]     = dDist[(base_row + tidy + 32) * n + (base_k + tidx + 32)];

        // D(k,b) into s + offset
        s[offset + tidy * TILE +  tidx ]                 = dDist[(base_k + tidy) * n + (base_j + tidx)];
        s[offset + tidy * TILE + (tidx + 32) ]           = dDist[(base_k + tidy) * n + (base_j + tidx + 32)];
        s[offset + (tidy + 32) * TILE +  tidx ]          = dDist[(base_k + tidy + 32) * n + (base_j + tidx)];
        s[offset + (tidy + 32) * TILE + (tidx + 32) ]    = dDist[(base_k + tidy + 32) * n + (base_j + tidx + 32)];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            int left0  = s[ tidy * TILE + k ];
            int left32 = s[(tidy + 32) * TILE + k ];
            int right0  = s[offset + k * TILE + tidx];
            int right32 = s[offset + k * TILE + (tidx + 32)];

            int via0 = left0 + right0; if (via0 < val0) val0 = via0;
            int via1 = left0 + right32; if (via1 < val1) val1 = via1;
            int via2 = left32 + right0; if (via2 < val2) val2 = via2;
            int via3 = left32 + right32; if (via3 < val3) val3 = via3;
        }

        // write back row tile
        dDist[(gi0      ) * n + (gj0      )] = val0;
        dDist[(gi0      ) * n + (gj0 + 32 )] = val1;
        dDist[(gi0 + 32 ) * n + (gj0      )] = val2;
        dDist[(gi0 + 32 ) * n + (gj0 + 32 )] = val3;
    } else {
        // column tile (b, r)
        b = blockIdx.y;
        if (b >= r) ++b;
        int base_i = b * TILE;
        int base_j = r * TILE;
        int base_k = r * TILE;

        int gi0 = base_i + tidy;
        int gj0 = base_j + tidx;

        int val0 = dDist[(gi0      ) * n + (gj0      )];
        int val1 = dDist[(gi0      ) * n + (gj0 + 32 )];
        int val2 = dDist[(gi0 + 32 ) * n + (gj0      )];
        int val3 = dDist[(gi0 + 32 ) * n + (gj0 + 32 )];

        // load D(b, r) and D(r, k)
        int offset = TILE * TILE;

        // D(b, r) -> s[0..]
        s[ tidy * TILE +  tidx ]                 = dDist[(base_i + tidy) * n + (base_k + tidx)];
        s[ tidy * TILE + (tidx + 32) ]           = dDist[(base_i + tidy) * n + (base_k + tidx + 32)];
        s[(tidy + 32) * TILE +  tidx ]           = dDist[(base_i + tidy + 32) * n + (base_k + tidx)];
        s[(tidy + 32) * TILE + (tidx + 32) ]     = dDist[(base_i + tidy + 32) * n + (base_k + tidx + 32)];

        // D(r, k) -> s + offset
        s[offset + tidy * TILE +  tidx ]                 = dDist[(base_k + tidy) * n + (base_j + tidx)];
        s[offset + tidy * TILE + (tidx + 32) ]           = dDist[(base_k + tidy) * n + (base_j + tidx + 32)];
        s[offset + (tidy + 32) * TILE +  tidx ]          = dDist[(base_k + tidy + 32) * n + (base_j + tidx)];
        s[offset + (tidy + 32) * TILE + (tidx + 32) ]    = dDist[(base_k + tidy + 32) * n + (base_j + tidx + 32)];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            int left0  = s[ tidy * TILE + k ];
            int left32 = s[(tidy + 32) * TILE + k ];
            int right0  = s[offset + k * TILE + tidx];
            int right32 = s[offset + k * TILE + (tidx + 32)];

            int via0 = left0 + right0; if (via0 < val0) val0 = via0;
            int via1 = left0 + right32; if (via1 < val1) val1 = via1;
            int via2 = left32 + right0; if (via2 < val2) val2 = via2;
            int via3 = left32 + right32; if (via3 < val3) val3 = via3;
        }

        // write back col tile
        dDist[(gi0      ) * n + (gj0      )] = val0;
        dDist[(gi0      ) * n + (gj0 + 32 )] = val1;
        dDist[(gi0 + 32 ) * n + (gj0      )] = val2;
        dDist[(gi0 + 32 ) * n + (gj0 + 32 )] = val3;
    }
}

__global__ void phase3_kernel(int r, int *dDist, int n) {
    __shared__ int s[2 * TILE * TILE];

    int tidx = threadIdx.x; // 0..31
    int tidy = threadIdx.y; // 0..31

    // skip row/col r by shifting indices
    int by = blockIdx.x;
    if (by >= r) ++by;
    int bx = blockIdx.y;
    if (bx >= r) ++bx;

    int base_i = by * TILE;
    int base_j = bx * TILE;
    int base_k = r  * TILE;

    int gi0 = base_i + tidy;
    int gj0 = base_j + tidx;

    int val0 = dDist[(gi0      ) * n + (gj0      )];
    int val1 = dDist[(gi0      ) * n + (gj0 + 32 )];
    int val2 = dDist[(gi0 + 32 ) * n + (gj0      )];
    int val3 = dDist[(gi0 + 32 ) * n + (gj0 + 32 )];

    int offset = TILE * TILE;

    // load D(by, r)
    s[ tidy * TILE +  tidx ]                 = dDist[(base_i + tidy) * n + (base_k + tidx)];
    s[ tidy * TILE + (tidx + 32) ]           = dDist[(base_i + tidy) * n + (base_k + tidx + 32)];
    s[(tidy + 32) * TILE +  tidx ]           = dDist[(base_i + tidy + 32) * n + (base_k + tidx)];
    s[(tidy + 32) * TILE + (tidx + 32) ]     = dDist[(base_i + tidy + 32) * n + (base_k + tidx + 32)];

    // load D(r, bx)
    s[offset + tidy * TILE +  tidx ]                 = dDist[(base_k + tidy) * n + (base_j + tidx)];
    s[offset + tidy * TILE + (tidx + 32) ]           = dDist[(base_k + tidy) * n + (base_j + tidx + 32)];
    s[offset + (tidy + 32) * TILE +  tidx ]          = dDist[(base_k + tidy + 32) * n + (base_j + tidx)];
    s[offset + (tidy + 32) * TILE + (tidx + 32) ]    = dDist[(base_k + tidy + 32) * n + (base_j + tidx + 32)];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        int left0  = s[ tidy * TILE + k ];
        int left32 = s[(tidy + 32) * TILE + k ];
        int right0  = s[offset + k * TILE + tidx];
        int right32 = s[offset + k * TILE + (tidx + 32)];

        int via0 = left0 + right0; if (via0 < val0) val0 = via0;
        int via1 = left0 + right32; if (via1 < val1) val1 = via1;
        int via2 = left32 + right0; if (via2 < val2) val2 = via2;
        int via3 = left32 + right32; if (via3 < val3) val3 = via3;
    }

    // write back
    dDist[(gi0      ) * n + (gj0      )] = val0;
    dDist[(gi0      ) * n + (gj0 + 32 )] = val1;
    dDist[(gi0 + 32 ) * n + (gj0      )] = val2;
    dDist[(gi0 + 32 ) * n + (gj0 + 32 )] = val3;
}
