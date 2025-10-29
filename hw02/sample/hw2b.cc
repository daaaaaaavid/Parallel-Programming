#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>  // SSE2 intrinsics
/*-------------------------------------------------------------
  PNG writer (same as before)
-------------------------------------------------------------*/
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/*-------------------------------------------------------------
  Compute a range of rows using OpenMP
-------------------------------------------------------------*/
void compute_rows(int start_row, int end_row, int width, int height, int max_iters,
                  double left, double right, double lower, double upper, int* buffer)
{
#pragma omp parallel for schedule(dynamic)
    for (int row = start_row; row < end_row; ++row) {
        double y0 = row * ((upper - lower) / height) + lower;

        // Precompute step in x-direction
        double dx = (right - left) / width;

        for (int x = 0; x < width; x += 2) {
            // Vector of two x0 values
            __m128d x0 = _mm_set_pd(left + (x + 1) * dx, left + x * dx);
            __m128d y0_vec = _mm_set1_pd(y0);

            __m128d a = _mm_setzero_pd();
            __m128d b = _mm_setzero_pd();
            __m128d two = _mm_set1_pd(2.0);
            __m128d four = _mm_set1_pd(4.0);

            int iters[2] = {0, 0};
            __m128d mask;

            for (int i = 0; i < max_iters; ++i) {
                // Compute a^2, b^2
                __m128d a2 = _mm_mul_pd(a, a);
                __m128d b2 = _mm_mul_pd(b, b);
                __m128d ab = _mm_mul_pd(a, b);

                // Compute |z|^2 = a^2 + b^2
                __m128d mag2 = _mm_add_pd(a2, b2);
                mask = _mm_cmplt_pd(mag2, four);

                // Early exit if both elements escaped
                int mask_bits = _mm_movemask_pd(mask);
                if (mask_bits == 0) break;

                // Update a and b where mask true
                __m128d a_new = _mm_add_pd(_mm_sub_pd(a2, b2), x0);
                __m128d b_new = _mm_add_pd(_mm_mul_pd(two, ab), y0_vec);

                a = _mm_or_pd(_mm_and_pd(mask, a_new), _mm_andnot_pd(mask, a));
                b = _mm_or_pd(_mm_and_pd(mask, b_new), _mm_andnot_pd(mask, b));

                // Increment iteration counters
                if (mask_bits & 0x1) iters[0]++;
                if (mask_bits & 0x2) iters[1]++;
            }

            // Store results back
            if (x < width) buffer[(row - start_row) * width + x] = iters[0];
            if (x + 1 < width) buffer[(row - start_row) * width + x + 1] = iters[1];
        }
    }
}

/*-------------------------------------------------------------
  Main program
-------------------------------------------------------------*/
int main(int argc, char** argv) {
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Divide rows among ranks */
    int rows_per_rank = height / size;
    int remainder = height % size;
    int start_row = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    int local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    int end_row = start_row + local_rows;

    /* Each rank only allocates its part */
    int* local_image = (int*)malloc(local_rows * width * sizeof(int));
    compute_rows(start_row, end_row, width, height, iters, left, right, lower, upper, local_image);

    /* Allocate buffer for root and zero-padding for Reduce */
    int* full_image = NULL;
    int* temp_image = (int*)calloc(width * height, sizeof(int));

    /* Copy local results into global-index positions */
    for (int i = 0; i < local_rows; ++i)
        memcpy(&temp_image[(start_row + i) * width], &local_image[i * width], width * sizeof(int));

    /* Combine all partial images */
    if (rank == 0)
        full_image = (int*)calloc(width * height, sizeof(int));

    MPI_Reduce(temp_image, full_image, width * height, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_png(filename, iters, width, height, full_image);
        free(full_image);
    }

    free(local_image);
    free(temp_image);
    MPI_Finalize();
    return 0;
}
