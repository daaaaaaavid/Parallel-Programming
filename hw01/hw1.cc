#include <cstdio>
#include <cstdlib>
#include <boost/sort/spreadsort/spreadsort.hpp>
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

    int *counts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));

    int active_procs = size;
    // if(active_procs > size) active_procs = size;

    int base = N / active_procs;
    int remainder = N % active_procs;
    int offset = 0;
    
    for (int i = 0; i < active_procs; i++) {
        counts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
    for (int i = active_procs; i < size; i++) {
        counts[i] = 0;   // unused ranks
        displs[i] = 0;
    }

    float *local_array = (float*) malloc((counts[rank]+1) * sizeof(float));
    float *local_array_tmp = (float*) malloc((counts[rank]+1) * sizeof(float));
    float *local_array_new = (float*) malloc((counts[rank]+1) * sizeof(float));

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY,
              MPI_INFO_NULL, &input_file);

    MPI_File_read_at(input_file,
                    displs[rank] * sizeof(float),  // byte offset
                    local_array,                   // where to store locally
                    counts[rank],                  // number of elements to read
                    MPI_FLOAT,
                    MPI_STATUS_IGNORE);

    int odd_even, i, fill_count, index1, index2;
	int sorted = 0, local_sorted=0;
    boost::sort::spreadsort::spreadsort(local_array, local_array+counts[rank]);
    while (!sorted) {
        local_sorted = 1;
        for (odd_even = 0; odd_even < 2; odd_even++){
            for (i = odd_even; i < active_procs - 1; i += 2){
                if(rank == i){
                    MPI_Sendrecv(local_array, counts[i], MPI_FLOAT, i+1, 0,
                    local_array_tmp, counts[i+1], MPI_FLOAT, i+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    fill_count = 0;
                    index1 = 0;
                    index2 = 0;
                    if(local_array[counts[i]-1] <= local_array_tmp[0]) continue;
                    while(fill_count<counts[i]){
                        if(index2 >= counts[i+1] || (index1 < counts[i] && local_array[index1] <= local_array_tmp[index2])){
                            local_array_new[fill_count] = local_array[index1];
                            index1++;
                        } else {
                            local_array_new[fill_count] = local_array_tmp[index2];
                            index2++;
                            local_sorted = 0;
                        }
                        fill_count++;
                    }
                    float *tmp = local_array;
                    local_array = local_array_new;
                    local_array_new = tmp;
                } else if(rank == i+1){
                    MPI_Sendrecv(local_array, counts[i+1], MPI_FLOAT, i, 0,
                    local_array_tmp, counts[i], MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    fill_count = counts[i+1]-1;
                    index1 = counts[i+1]-1;
                    index2 = counts[i]-1;
                    if(local_array[0] >= local_array_tmp[counts[i]-1]) continue;
                    while(fill_count>=0){
                        if(index2 < 0 || (index1 >=0 && local_array[index1]>=local_array_tmp[index2])){
                            local_array_new[fill_count] = local_array[index1];
                            index1--;
                        } else {
                            local_array_new[fill_count] = local_array_tmp[index2];
                            index2--;
                            local_sorted = 0;
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

    MPI_File_open(MPI_COMM_WORLD, output_filename,
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &output_file);

    MPI_File_write_at(output_file,
                    displs[rank] * sizeof(float),
                    local_array,
                    counts[rank],
                    MPI_FLOAT,
                    MPI_STATUS_IGNORE);

    MPI_File_close(&output_file);

    free(local_array);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

