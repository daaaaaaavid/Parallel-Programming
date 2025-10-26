#include <cstdio>
#include <cstdlib>

#include <mpi.h>

void swap(float *a, float *b){
    float temp = *a;
    *a = *b;
    *b = temp;
}
int divid(float arr[], int low, int high){
    float pivot = arr[high];
    int i = low-1;
    for(int j=low;j<high;j++){
        if(arr[j] <= pivot){
            i++;
            swap(&arr[j], &arr[i]);
        }
    }
    swap(&arr[i+1], &arr[high]);
    return i+1;
}
void quicksort(float arr[], int low, int high){
    if(low<high){
        int p = divid(arr, low, high);
        quicksort(arr, low, p-1);
        quicksort(arr, p+1, high);
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);                  // size of array

    int bound;
    if(N<size) bound = N;
    else bound = size;

    const char *input_filename  = argv[2];  // input file
    const char *output_filename = argv[3];  // output file
    
    MPI_File input_file, output_file;
    // --- Compute counts/displacements ---
    int base = N / size;
    int remainder = N % size;

    int *counts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }

    // --- Allocate buffers ---
    float *global_array = NULL;
    if (rank == 0) global_array = (float*) malloc(N * sizeof(float));
    float *local_array = (float*) malloc(counts[rank] * sizeof(float));

    // --- Read input file (only rank 0) ---
    if (rank == 0) {

        MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &input_file);

        MPI_File_read_at(input_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);

        MPI_File_close(&input_file);
    }

    // --- Scatter data to processes ---
    MPI_Scatterv(global_array, counts, displs, MPI_FLOAT,
                 local_array, counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    int odd_even, i;
    float temp1, temp2;
	int sorted = 0, local_sorted=0;
    
    // printf("rank %d get %f\n", rank, local_array[0]);
    // printf("Total process %d\n", N);
    while (!sorted) {
        quicksort(local_array, 0, counts[rank]-1);
        local_sorted = 1;
        for (odd_even = 0; odd_even < 2; odd_even++){
            for (i = odd_even; i < bound - 1; i += 2){
                if (rank == i) {
                    float num1;
                    MPI_Request reqs[2];
                    MPI_Status stats[2];

                    // Non-blocking send and recv
                    MPI_Isend(&local_array[counts[rank] - 1], 1, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD, &reqs[0]);
                    MPI_Irecv(&num1, 1, MPI_FLOAT, i + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[1]);

                    // Wait for both to finish
                    MPI_Waitall(2, reqs, stats);

                    if (local_array[counts[rank] - 1] > num1) {
                        local_array[counts[rank] - 1] = num1;
                        local_sorted = 0;
                    }

                } else if (rank == i + 1) {
                    float num2;
                    MPI_Request reqs[2];
                    MPI_Status stats[2];

                    // Non-blocking recv and send
                    MPI_Irecv(&num2, 1, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[0]);
                    MPI_Isend(&local_array[0], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs[1]);

                    // Wait for both to finish
                    MPI_Waitall(2, reqs, stats);

                    if (local_array[0] < num2) {
                        local_array[0] = num2;
                        local_sorted = 0;
                    }
                }
            }
        }
        MPI_Allreduce(&local_sorted, &sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    MPI_Gatherv(local_array, counts[rank], MPI_FLOAT,
                global_array, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // // --- Write output file (only rank 0) ---
    if (rank == 0) {

        MPI_File_open(MPI_COMM_SELF, output_filename,
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

        MPI_File_write_at(output_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);

        MPI_File_close(&output_file);

        // printf("Processed array written to %s\n", output_filename);
    }

    // // cleanup
    free(local_array);
    free(counts);
    free(displs);
    if (rank == 0) free(global_array);

    // MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    // MPI_File_write_at(output_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

