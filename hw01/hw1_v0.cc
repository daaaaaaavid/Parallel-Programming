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
    float data[1];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    int odd_even, i;
    float temp1, temp2;
	int sorted = 0, local_sorted=0;
    MPI_Status status;
    // printf("Total process %d\n", N);
    while (!sorted) {
        local_sorted = 1;
        for (odd_even = 0; odd_even < 2; odd_even++){
            for (i = odd_even; i < N - 1; i += 2){
                if(rank == i){
                    float num1;
                    MPI_Send(&data[0], 1, MPI_FLOAT, i+1, 0, MPI_COMM_WORLD);
                    MPI_Recv(&num1, 1, MPI_FLOAT, i+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[0] > num1){
                        printf("%d receive %f from %d: %d\n", i, num1, i+1, sorted);
                        data[0] = num1;
                        local_sorted = 0;
                    }
                } else if(rank == i+1){
                    float num2;
                    MPI_Recv(&num2, 1, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(&data[0], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    if(data[0] < num2){
                        printf("%d receive %f from %d: %d\n", i+1, num2, i, sorted);
                        data[0] = num2;
                        local_sorted = 0;
                    }
                }
            }
        }
        MPI_Allreduce(&local_sorted, &sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

