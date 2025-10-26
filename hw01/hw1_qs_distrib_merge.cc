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
    float *local_array_tmp = (float*) malloc((counts[0]+1) * sizeof(float));
    float *local_array_new = (float*) malloc(counts[rank] * sizeof(float));
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

    int odd_even, i, fill_count, index1, index2;
	int sorted = 0, local_sorted=0;
    
    // printf("rank %d get %f\n", rank, local_array[0]);
    // printf("Total process %d\n", N);
    while (!sorted) {
        quicksort(local_array, 0, counts[rank]-1);
        local_sorted = 1;
        for (odd_even = 0; odd_even < 2; odd_even++){
            for (i = odd_even; i < bound - 1; i += 2){
                if(rank == i){
                    MPI_Send(&local_array, counts[i], MPI_FLOAT, i+1, 0, MPI_COMM_WORLD);
                    MPI_Recv(&local_array_tmp, counts[i+1], MPI_FLOAT, i+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    fill_count = 0;
                    index1 = 0;
                    index2 = 0;
                    while(fill_count<counts[i]){
                        if(local_array[index1]<=local_array_tmp[index2] || index2 >= counts[i+1]){
                            local_array_new[fill_count] = local_array[index1];
                            index1++;
                        } else {
                            local_array_new[fill_count] = local_array_tmp[index2];
                            index2++;
                        }
                        fill_count++;
                    }
                    float *tmp = local_array;
                    local_array = local_array_new;
                    local_array_new = tmp;
                } else if(rank == i+1){
                    MPI_Recv(&local_array_tmp, counts[i], MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(&local_array, counts[i+1], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    fill_count = counts[i+1]-1;
                    index1 = counts[i+1]-1;
                    index2 = counts[i]-1;
                    while(fill_count>=0){
                        if(local_array[index1]>local_array_tmp[index2] || index2<0){
                            local_array_new[fill_count] = local_array[index1];
                            index1--;
                        } else {
                            local_array_new[fill_count] = local_array_tmp[index2];
                            index2--;
                        }
                        fill_count--;
                    }
                    float *tmp = local_array;
                    local_array = local_array_new;
                    local_array_new = tmp;
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

