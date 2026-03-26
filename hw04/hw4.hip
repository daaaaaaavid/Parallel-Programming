// flash_attention_hip_worstcase.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <hip/hip_runtime.h>

#define TILE_N_R 64   // Br = threads per block
#define TILE_N_C 32   // Bc = keys per tile
#define MAX_D 128     // worst-case embedding dimension

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------- Kernel ----------------
__global__ void flashAttentionKernelHIP(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d,
    int Br, int Bc,
    int Tr, int Tc,
    float scale
){
    extern __shared__ float smem[]; // Ks + Vs
    float* Ks = smem;
    float* Vs = Ks + Bc*d;

    const int batch = blockIdx.z;
    const int br = blockIdx.x;
    const int row_start = br * Br;
    const int ti = threadIdx.x;
    const int row = row_start + ti;
    if (row >= N) return;

    // Load Q row into registers
    float Q_row[MAX_D];
    for (int k = 0; k < d; ++k)
        Q_row[k] = Q[(batch*N + row)*d + k];

    float acc[MAX_D]; // accumulator
    for (int k = 0; k < d; ++k) acc[k] = 0.f;

    float m = -FLT_MAX;
    float l = 0.f;

    for (int bc = 0; bc < Tc; ++bc) {
        int col_start = bc * Bc;
        int tile_size = min(Bc, N - col_start);

        // load K/V tile into shared memory
        for (int i = ti; i < tile_size*d; i += blockDim.x) {
            int r = i / d;
            int c = i % d;
            Ks[r*d + c] = K[(batch*N + col_start + r)*d + c];
            Vs[r*d + c] = V[(batch*N + col_start + r)*d + c];
        }
        __syncthreads();

        // compute scores
        float S_row[TILE_N_C];
        float tile_max = -FLT_MAX;
        for (int j = 0; j < tile_size; ++j) {
            float dot = 0.f;
            for (int k = 0; k < d; ++k)
                dot += Q_row[k] * Ks[j*d + k];
            S_row[j] = dot * scale;
            tile_max = fmaxf(tile_max, S_row[j]);
        }

        // online softmax update
        float new_m = fmaxf(m, tile_max);
        float alpha = (m == -FLT_MAX) ? 1.f : expf(m - new_m);
        for (int k = 0; k < d; ++k) acc[k] *= alpha;
        l *= alpha;

        for (int j = 0; j < tile_size; ++j) {
            float w = expf(S_row[j] - new_m);
            l += w;
            for (int k = 0; k < d; ++k)
                acc[k] += w * Vs[j*d + k];
        }
        m = new_m;
    }

    // write output
    float inv_l = 1.f / (l + 1e-9f);
    for (int k = 0; k < d; ++k)
        O[(batch*N + row)*d + k] = acc[k] * inv_l;
}

// ---------------- Host I/O ----------------
void read_input(const char* filename, int &B, int &N, int &d,
                float* &Q, float* &K, float* &V) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen input"); exit(1); }
    fread(&B, sizeof(int), 1, f);
    fread(&N, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);

    if (d > MAX_D) { fprintf(stderr, "Embedding dimension exceeds MAX_D=%d\n", MAX_D); exit(1); }

    size_t elems_per_batch = (size_t)N*d;
    size_t size = (size_t)B*elems_per_batch;

    Q = (float*)malloc(size*sizeof(float));
    K = (float*)malloc(size*sizeof(float));
    V = (float*)malloc(size*sizeof(float));

    for (int b=0; b<B; ++b) {
        fread(Q + b*elems_per_batch, sizeof(float), elems_per_batch, f);
        fread(K + b*elems_per_batch, sizeof(float), elems_per_batch, f);
        fread(V + b*elems_per_batch, sizeof(float), elems_per_batch, f);
    }
    fclose(f);
}

void write_output(const char* filename, int B, int N, int d, float* O) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen output"); exit(1); }
    fwrite(O, sizeof(float), (size_t)B*N*d, f);
    fclose(f);
}

// ---------------- Main ----------------
int main(int argc, char* argv[]) {
    if (argc != 3) { printf("Usage: %s <input> <output>\n", argv[0]); return 1; }

    int B, N, d;
    float *Q_h, *K_h, *V_h;
    read_input(argv[1], B, N, d, Q_h, K_h, V_h);

    size_t size_bytes = (size_t)B*N*d*sizeof(float);
    float* O_h = (float*)malloc(size_bytes);

    float *Q_d, *K_d, *V_d, *O_d;
    hipMalloc(&Q_d, size_bytes);
    hipMalloc(&K_d, size_bytes);
    hipMalloc(&V_d, size_bytes);
    hipMalloc(&O_d, size_bytes);

    hipMemcpy(Q_d, Q_h, size_bytes, hipMemcpyHostToDevice);
    hipMemcpy(K_d, K_h, size_bytes, hipMemcpyHostToDevice);
    hipMemcpy(V_d, V_h, size_bytes, hipMemcpyHostToDevice);

    int Br = TILE_N_R; // 64 threads per block
    int Bc = TILE_N_C; // 32 keys per tile
    int Tr = ceil_div(N, Br);
    int Tc = ceil_div(N, Bc);
    dim3 grid(Tr, 1, B), block(Br);

    size_t shared_mem_size = (size_t)Bc*d*2*sizeof(float); // Ks + Vs
    float scale = 1.f / sqrtf((float)d);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    hipLaunchKernelGGL(flashAttentionKernelHIP,
                       grid, block, shared_mem_size, 0,
                       Q_d, K_d, V_d, O_d, N, d, Br, Bc, Tr, Tc, scale);

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.f;
    hipEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", ms);

    hipMemcpy(O_h, O_d, size_bytes, hipMemcpyDeviceToHost);
    write_output(argv[2], B, N, d, O_h);

    free(Q_h); free(K_h); free(V_h); free(O_h);
    hipFree(Q_d); hipFree(K_d); hipFree(V_d); hipFree(O_d);

    return 0;
}
