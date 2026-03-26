// flash_attention_tiled_vectorized_optimized.cu
// Optimized version of flashAttentionKernel_v2
// - Moves Ms/Ls into registers (per-thread)
// - Removes shared Ss (scores) and per-row shared Ms/Ls
// - Stores per-thread scores in registers (stack) to avoid shared mem
// - Keeps Q/K/V tiles in shared memory and vectorized dot using float4
// - Removes writes to global L and M for improved performance

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define TILE_N_R 64
#define TILE_N_C 32
#define MAX_D 128

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------- Optimized FlashAttention Kernel ----------------
__global__ void flashAttentionKernel_opt(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d,
    int Br, int Bc,
    int Tr, int Tc,
    float scale
){
    extern __shared__ float smem[];

    // Layout: Qs [Br*d], Ks [Bc*d], Vs [Bc*d]
    float* Qs = smem;                      // [Br][d]
    float* Ks = Qs + Br * d;               // [Bc][d]
    float* Vs = Ks + Bc * d;               // [Bc][d]

    const int batch = blockIdx.z;
    const int br = blockIdx.x;
    const int row_start = br * Br;
    const int ti = threadIdx.x;
    const int row = row_start + ti;

    if (row >= N) return; // out-of-range

    // Load Q row into shared memory (cooperative but each thread its own row)
    for (int k = 0; k < d; ++k) {
        Qs[ti * d + k] = Q[(batch * N + row) * d + k];
    }

    __syncthreads(); // ensure Qs is visible before using with Ks

    // per-thread running softmax scalars in registers
    float m = -FLT_MAX; // running max
    float l = 0.f;      // running sum-of-exp

    // accumulator for output (in registers)
    float acc[MAX_D];
#pragma unroll
    for (int k = 0; k < d; ++k) acc[k] = 0.f;

    const int d4 = d / 4; // d % 4 == 0 guaranteed by host

    // per-thread temporary array to hold tile scores
    // Bc is compile-time constant (TILE_N_C)
    float S_row[TILE_N_C];

    // iterate over key tiles
    for (int bc = 0; bc < Tc; ++bc) {
        const int col_start = bc * Bc;

        // load K, V tile cooperatively into shared memory
        // each thread strides over Bc*d elements
        for (int i = ti; i < Bc * d; i += blockDim.x) {
            int r = i / d; // which key within tile
            int c = i % d; // which channel
            int col = col_start + r;
            float kval = (col < N) ? K[(batch * N + col) * d + c] : 0.f;
            float vval = (col < N) ? V[(batch * N + col) * d + c] : 0.f;
            Ks[r * d + c] = kval;
            Vs[r * d + c] = vval;
        }

        __syncthreads(); // ensure Ks/Vs visible to all threads

        // Compute tile scores for this query row into register array S_row
        float* Q_row = &Qs[ti * d];

        for (int j = 0; j < Bc; ++j) {
            float dot = 0.f;
            float4* K_row4 = (float4*)(&Ks[j * d]);
            float4* Q_row4 = (float4*)Q_row;
            #pragma unroll
            for (int kk = 0; kk < d4; ++kk) {
                float4 kv = K_row4[kk];
                float4 qv = Q_row4[kk];
                dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
            }
            S_row[j] = dot * scale;
        }

        // compute tile max in registers (per-thread)
        float tile_max = -FLT_MAX;
        for (int j = 0; j < Bc; ++j) tile_max = fmaxf(tile_max, S_row[j]);

        // online softmax combine
        float new_m = fmaxf(m, tile_max);
        float alpha = (m == -FLT_MAX) ? 1.f : expf(m - new_m);

        // rescale previous accumulators
        for (int k = 0; k < d; ++k) acc[k] *= alpha;
        l *= alpha;

        // accumulate contributions from this tile
        for (int j = 0; j < Bc; ++j) {
            int col = col_start + j;
            if (col >= N) continue; // guard tail keys
            float w = expf(S_row[j] - new_m);
            l += w;
            float* V_row = &Vs[j * d];
            for (int k = 0; k < d; ++k) acc[k] += w * V_row[k];
        }

        m = new_m;
        __syncthreads(); // ensure next tile load safe (Ks/Vs reuse)
    }

    // write output (normalize by l)
    float inv_l = 1.f / (l + 1e-9f);
    float* out = &O[(batch * N + row) * d];
    for (int k = 0; k < d; ++k) out[k] = acc[k] * inv_l;
}

// ---------------- Host I/O (unchanged except removed L/M) ----------------
void read_input(const char* filename, int &B, int &N, int &d,
                float* &Q, float* &K, float* &V) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen input"); exit(1); }

    fread(&B, sizeof(int), 1, f);
    fread(&N, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);

    if (d % 4 != 0) { fprintf(stderr, "Embedding dimension must be divisible by 4\n"); exit(1); }
    if (d > MAX_D) { fprintf(stderr, "Embedding dimension d=%d exceeds MAX_D=%d\n", d, MAX_D); exit(1); }

    size_t elems_per_batch = (size_t)N * d;
    size_t size = (size_t)B * elems_per_batch;
    Q = (float*)malloc(size * sizeof(float));
    K = (float*)malloc(size * sizeof(float));
    V = (float*)malloc(size * sizeof(float));

    for (int b = 0; b < B; ++b) {
        fread(Q + (size_t)b * elems_per_batch, sizeof(float), elems_per_batch, f);
        fread(K + (size_t)b * elems_per_batch, sizeof(float), elems_per_batch, f);
        fread(V + (size_t)b * elems_per_batch, sizeof(float), elems_per_batch, f);
    }
    fclose(f);
}

void write_output(const char* filename, int B, int N, int d, float* O) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen output"); exit(1); }
    fwrite(O, sizeof(float), (size_t)B * N * d, f);
    fclose(f);
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    if (argc != 3) { printf("Usage: %s <input> <output>\n", argv[0]); return 1; }

    int B, N, d;
    float *Q_h, *K_h, *V_h;
    read_input(argv[1], B, N, d, Q_h, K_h, V_h);

    size_t size_bytes = (size_t)B * N * d * sizeof(float);
    float *O_h = (float*)malloc(size_bytes);

    float *Q_d, *K_d, *V_d, *O_d;
    cudaMalloc(&Q_d, size_bytes);
    cudaMalloc(&K_d, size_bytes);
    cudaMalloc(&V_d, size_bytes);
    cudaMalloc(&O_d, size_bytes);

    cudaMemcpy(Q_d, Q_h, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, size_bytes, cudaMemcpyHostToDevice);

    int Br = TILE_N_R, Bc = TILE_N_C;
    int Tr = ceil_div(N, Br), Tc = ceil_div(N, Bc);
    int threads = Br;
    dim3 grid(Tr, 1, B), block(threads);

    // shared mem: (Br + 2*Bc) * d * sizeof(float)
    size_t sram_size = (size_t)(Br + 2 * Bc) * d * sizeof(float);
    float scale = 1.f / sqrtf((float)d);

    flashAttentionKernel_opt<<<grid, block, sram_size>>>(
        Q_d, K_d, V_d, O_d, N, d, Br, Bc, Tr, Tc, scale
    );

    cudaDeviceSynchronize();

    cudaMemcpy(O_h, O_d, size_bytes, cudaMemcpyDeviceToHost);
    write_output(argv[2], B, N, d, O_h);

    free(Q_h); free(K_h); free(V_h); free(O_h);
    cudaFree(Q_d); cudaFree(K_d);
    cudaFree(V_d); cudaFree(O_d);

    return 0;
}
