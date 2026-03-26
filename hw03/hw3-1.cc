#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 512;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {

        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);

    }
}

// void block_FW(int B) {
//     int round = (n + B - 1) / B;

//     for (int r = 0; r < round; ++r) {

//         // Phase 1 (sequential)
//         cal(B, r, r, r, 1, 1);

//         // ======================
//         // Phase 2 (parallel)
//         // ======================
//         #pragma omp parallel sections
//         {
//             #pragma omp section
//             cal(B, r, r, 0, r, 1);

//             #pragma omp section
//             cal(B, r, r, r+1, round-r-1, 1);

//             #pragma omp section
//             cal(B, r, 0, r, 1, r);

//             #pragma omp section
//             cal(B, r, r+1, r, 1, round-r-1);
//         }

//         // ======================
//         // Phase 3 (parallel)
//         // ======================
//         #pragma omp parallel sections
//         {
//             #pragma omp section
//             cal(B, r, 0, 0, r, r);

//             #pragma omp section
//             cal(B, r, 0, r+1, round-r-1, r);

//             #pragma omp section
//             cal(B, r, r+1, 0, r, round-r-1);

//             #pragma omp section
//             cal(B, r, r+1, r+1, round-r-1, round-r-1);
//         }
//     }
// }

void cal(int B, int Round,
         int block_start_x, int block_start_y,
         int block_width, int block_height)
{
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    // parallelize block-level work
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {

            int xs = b_i * B;
            int xe = std::min(xs + B, n);
            int ys = b_j * B;
            int ye = std::min(ys + B, n);

            // sequential k loop (MANDATORY)
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {

                // OPTIONAL: parallelize (i,j) if B is large
                // #pragma omp parallel for collapse(2)
                for (int i = xs; i < xe; ++i) {
                    for (int j = ys; j < ye; ++j) {
                        int via = Dist[i][k] + Dist[k][j];
                        if (via < Dist[i][j]) {
                            Dist[i][j] = via;
                        }
                    }
                }
            }
        }
    }
}
