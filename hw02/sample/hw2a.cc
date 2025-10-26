#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
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

typedef struct {
    int* shared_row;
    pthread_mutex_t row_lock;
    int start_row;
    int end_row;
    int width;
    int height;
    int max_iters;
    double left, right, lower, upper;
    int *image;

    struct timeval start_time;
    struct timeval end_time;
} thread_arg_t;

void* mandelbrot_worker(void* arg) {
    thread_arg_t* t = (thread_arg_t*)arg;
    gettimeofday(&t->start_time, NULL);

    int row;
    while (1) {
        // 取 row（保護共享變數）
        pthread_mutex_lock(&t->row_lock);
        row = *(t->shared_row);
        (*(t->shared_row))++;
        pthread_mutex_unlock(&t->row_lock);

        if (row >= t->height) break;

        double y0 = row * ((t->upper - t->lower) / t->height) + t->lower;

        for (int i = 0; i < t->width; ++i) {
            double x0 = i * ((t->right - t->left) / t->width) + t->left;
            int repeats = 0;
            double x = 0, y = 0, length_squared = 0;
            while (repeats < t->max_iters && length_squared < 4.0) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            t->image[row * t->width + i] = repeats;
        }
    }

    gettimeofday(&t->end_time, NULL);
    pthread_exit(NULL);
}


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

    /* detect CPU cores */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set);
    int nthreads = CPU_COUNT(&cpu_set);
    printf("Using %d threads\n", nthreads);

    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[nthreads];
    thread_arg_t args[nthreads];

    int rows_per_thread = height / nthreads;
    int next_row = 0;
    pthread_mutex_t row_lock = PTHREAD_MUTEX_INITIALIZER;

    for (int t = 0; t < nthreads; ++t) {
        args[t].shared_row = &next_row;
        args[t].row_lock = row_lock;  // 傳入鎖
        args[t].start_row = t * rows_per_thread;
        args[t].end_row = (t == nthreads - 1) ? height : (t + 1) * rows_per_thread;
        args[t].width = width;
        args[t].height = height;
        args[t].max_iters = iters;
        args[t].left = left;
        args[t].right = right;
        args[t].lower = lower;
        args[t].upper = upper;
        args[t].image = image;

        pthread_create(&threads[t], NULL, mandelbrot_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; ++t) {
        pthread_join(threads[t], NULL);

        double start = args[t].start_time.tv_sec + args[t].start_time.tv_usec / 1000000.0;
        double end = args[t].end_time.tv_sec + args[t].end_time.tv_usec / 1000000.0;
        double elapsed = end - start;

        printf("Thread %d execution time: %f seconds\n", t, elapsed);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}