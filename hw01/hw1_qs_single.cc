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

    const char *input_filename  = argv[2];  // input file
    const char *output_filename = argv[3];  // output file
    
    MPI_File input_file, output_file;

    float *global_array = NULL;
    if (rank == 0) global_array = (float*) malloc(N * sizeof(float));

    if (rank == 0) {

        MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &input_file);

        MPI_File_read_at(input_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);

        MPI_File_close(&input_file);

        quicksort(global_array, 0, N-1);
    }

    

    if (rank == 0) {

        MPI_File_open(MPI_COMM_SELF, output_filename,
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

        MPI_File_write_at(output_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);

        MPI_File_close(&output_file);

        // printf("Processed array written to %s\n", output_filename);
    }


    if (rank == 0) free(global_array);

    // MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    // MPI_File_write_at(output_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

