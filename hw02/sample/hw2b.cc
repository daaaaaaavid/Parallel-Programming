#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h> // SSE2 intrinsics

#define TAG_WORK 1
#define TAG_RESULT 2
#define TAG_STOP 3

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
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

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
  Mandelbrot computation with SSE2 vectorization
-------------------------------------------------------------*/
void compute_row(int row, int width, int height, int max_iters,
                 double left, double right, double lower, double upper, int* row_data)
{
    double y0 = row * ((upper - lower) / height) + lower;
    double dx = (right - left) / width;

#pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < width; x += 2) {
        __m128d x0 = _mm_set_pd(left + (x + 1) * dx, left + x * dx);
        __m128d y0_vec = _mm_set1_pd(y0);

        __m128d a = _mm_setzero_pd();
        __m128d b = _mm_setzero_pd();
        __m128d two = _mm_set1_pd(2.0);
        __m128d four = _mm_set1_pd(4.0);

        int iters[2] = {0, 0};

        for (int i = 0; i < max_iters; ++i) {
            __m128d a2 = _mm_mul_pd(a, a);
            __m128d b2 = _mm_mul_pd(b, b);
            __m128d ab = _mm_mul_pd(a, b);

            __m128d mag2 = _mm_add_pd(a2, b2);
            __m128d mask = _mm_cmplt_pd(mag2, four);
            int mask_bits = _mm_movemask_pd(mask);

            if (mask_bits == 0)
                break;

            __m128d a_new = _mm_add_pd(_mm_sub_pd(a2, b2), x0);
            __m128d b_new = _mm_add_pd(_mm_mul_pd(two, ab), y0_vec);

            a = _mm_or_pd(_mm_and_pd(mask, a_new), _mm_andnot_pd(mask, a));
            b = _mm_or_pd(_mm_and_pd(mask, b_new), _mm_andnot_pd(mask, b));

            if (mask_bits & 0x1) iters[0]++;
            if (mask_bits & 0x2) iters[1]++;
        }

        // Store results
        if (x < width) row_data[x] = iters[0];
        if (x + 1 < width) row_data[x + 1] = iters[1];
    }
}

/*-------------------------------------------------------------
  Main program (MPI + OpenMP hybrid)
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

    if (rank == 0) {
        int *image = (int*)malloc(width * height * sizeof(int));
        int next_row = 0;
        MPI_Status status;

        // send initial rows
        for (int worker = 1; worker < size; ++worker) {
            if (next_row < height) {
                MPI_Send(&next_row, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
            }
        }

        // receive + dispatch
        while (next_row < height) {
            int row_index;
            MPI_Recv(&row_index, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

            int* row_buffer = (int*)malloc(width * sizeof(int));
            MPI_Recv(row_buffer, width, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&image[row_index * width], row_buffer, width * sizeof(int));
            free(row_buffer);

            MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
            next_row++;
        }

        // finalize remaining rows
        for (int worker = 1; worker < size; ++worker) {
            int row_index;
            MPI_Recv(&row_index, 1, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, &status);

            int* row_buffer = (int*)malloc(width * sizeof(int));
            MPI_Recv(row_buffer, width, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&image[row_index * width], row_buffer, width * sizeof(int));
            free(row_buffer);
        }

        for (int worker = 1; worker < size; ++worker)
            MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);

        write_png(filename, iters, width, height, image);
        free(image);
    } else {
        MPI_Status status;
        int row_index;
        int* row_data = (int*)malloc(width * sizeof(int));

        while (1) {
            MPI_Recv(&row_index, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP)
                break;

            compute_row(row_index, width, height, iters, left, right, lower, upper, row_data);

            MPI_Send(&row_index, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(row_data, width, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        free(row_data);
    }

    MPI_Finalize();
    return 0;
}
