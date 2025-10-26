#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);                  // size of array
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
    
    printf("rank %d get %f\n", rank, local_array[0]);

    MPI_Gatherv(local_array, counts[rank], MPI_FLOAT,
                global_array, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    free(local_array);
    free(counts);
    free(displs);
    if (rank == 0) free(global_array);


    MPI_Finalize();
    return 0;
}

