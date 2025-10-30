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
#include <immintrin.h>  // for SSE2 intrinsics

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
  Mandelbrot computation for a single row (OpenMP inside)
-------------------------------------------------------------*/
void compute_row(int row, int width, int height, int max_iters,
                 double left, double right, double lower, double upper, int* row_data)
{
    double y0 = row * ((upper - lower) / height) + lower;

    int idx = 0;
    int CHUNK = width / 30 / omp_get_max_threads();        // adjustable chunk size
    if (CHUNK < 2) CHUNK = 2;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        __m128d FOUR = _mm_set1_pd(4.0);

        while (1) {
            int start, end;

            // Acquire a chunk of work
            omp_set_lock(&lock);
            if (idx >= width) {
                omp_unset_lock(&lock);
                break;
            }
            start = idx;
            end = idx + CHUNK;
            if (end > width) end = width;
            idx = end;
            omp_unset_lock(&lock);

            // Compute each pixel (SIMD for pairs)
            for (int i = start; i < end; i += 2) {
                // Handle last pixel if odd width
                if (i + 1 >= end) {
                    double x0 = i * ((right - left) / width) + left;
                    int repeats = 0;
                    double x = 0, y = 0, length_squared = 0;
                    while (repeats < max_iters && length_squared < 4.0) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    row_data[i] = repeats;
                    continue;
                }

                // Vectorized version for 2 pixels
                double x0_arr[2] = {
                    i * ((right - left) / width) + left,
                    (i + 1) * ((right - left) / width) + left
                };

                __m128d x0_vec = _mm_loadu_pd(x0_arr);
                __m128d x = _mm_set1_pd(0.0);
                __m128d y = _mm_set1_pd(0.0);
                __m128d length_sq = _mm_set1_pd(0.0);

                int repeats[2] = {0, 0};
                int done_mask = 0;

                for (int k = 0; k < max_iters && done_mask != 3; ++k) {
                    __m128d x2 = _mm_mul_pd(x, x);
                    __m128d y2 = _mm_mul_pd(y, y);
                    __m128d xy = _mm_mul_pd(x, y);

                    __m128d temp_x = _mm_add_pd(_mm_sub_pd(x2, y2), x0_vec);
                    y = _mm_add_pd(_mm_add_pd(xy, xy), _mm_set1_pd(y0));
                    x = temp_x;

                    length_sq = _mm_add_pd(x2, y2);

                    // Compare: length_sq < 4.0
                    __m128d mask = _mm_cmplt_pd(length_sq, FOUR);
                    int active = _mm_movemask_pd(mask);

                    // Increment counts only for active pixels
                    if (active & 1) ++repeats[0];
                    if (active & 2) ++repeats[1];
                    done_mask = (~active) & 3;
                }

                row_data[i] = repeats[0];
                row_data[i + 1] = repeats[1];
            }
        }
    }

    omp_destroy_lock(&lock);
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

    // if (rank == 0) {
    //     printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, omp_get_max_threads());
    // }

    /* ---------------- Master process ---------------- */
    if (rank == 0) {
        int *image = (int*)malloc(width * height * sizeof(int));
        int next_row = 0;
        MPI_Status status;

        // send initial rows to each worker
        for (int worker = 1; worker < size; ++worker) {
            if (next_row < height) {
                MPI_Send(&next_row, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
            }
        }

        // receive results and assign new rows dynamically
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

        // gather final rows
        for (int worker = 1; worker < size; ++worker) {
            int row_index;
            MPI_Recv(&row_index, 1, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, &status);

            int* row_buffer = (int*)malloc(width * sizeof(int));
            MPI_Recv(row_buffer, width, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&image[row_index * width], row_buffer, width * sizeof(int));
            free(row_buffer);
        }

        // tell all workers to stop
        for (int worker = 1; worker < size; ++worker)
            MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);

        // output PNG
        write_png(filename, iters, width, height, image);
        free(image);
    }

    /* ---------------- Worker processes ---------------- */
    else {
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