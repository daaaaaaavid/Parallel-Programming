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

    int active_procs = 24;
    if(active_procs > size) active_procs = size;

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

    double start_time, end_time, tc1, tc2, t1, t2;
    double io_time = 0.0, comm_time = 0.0, comp_time = 0.0;
    start_time = MPI_Wtime();

    // t1 = MPI_Wtime();
    // MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY,
    //           MPI_INFO_NULL, &input_file);

    // MPI_File_read_at(input_file,
    //                 displs[rank] * sizeof(float),  // byte offset
    //                 local_array,                   // where to store locally
    //                 counts[rank],                  // number of elements to read
    //                 MPI_FLOAT,
    //                 MPI_STATUS_IGNORE);

    // t2 = MPI_Wtime();
    // io_time += (t2 - t1);

    float *global_array = NULL;
    if (rank == 0) global_array = (float*) malloc(N * sizeof(float));

    // --- Read input file (only rank 0) ---
    if (rank == 0) {
        t1 = MPI_Wtime();
        MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &input_file);

        MPI_File_read_at(input_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);

        MPI_File_close(&input_file);
        t2 = MPI_Wtime();
        io_time += (t2 - t1);
    }

    tc1 = MPI_Wtime();
    MPI_Scatterv(global_array, counts, displs, MPI_FLOAT,
                 local_array, counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    tc2 = MPI_Wtime();
    comm_time += (tc2 - tc1);

    int odd_even, i, fill_count, index1, index2;
	int sorted = 0, local_sorted=0, count=0;

    double t3 = MPI_Wtime();
    boost::sort::spreadsort::spreadsort(local_array, local_array + counts[rank]);
    double t4 = MPI_Wtime();
    comp_time += (t4 - t3);

    // while (!sorted) {
    for(int k=0;k<active_procs+2;k++){
        local_sorted = 1;
        for (odd_even = 0; odd_even < 2; odd_even++){
            for (i = odd_even; i < active_procs - 1; i += 2){
                if(rank == i){
                    // MPI_Send(local_array, counts[i], MPI_FLOAT, i+1, 0, MPI_COMM_WORLD);
                    // MPI_Recv(local_array_tmp, counts[i+1], MPI_FLOAT, i+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    tc1 = MPI_Wtime();
                    MPI_Sendrecv(local_array, counts[i], MPI_FLOAT, i+1, 0,
                    local_array_tmp, counts[i+1], MPI_FLOAT, i+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    tc2 = MPI_Wtime();
                    comm_time += (tc2 - tc1);
                    
                    t3 = MPI_Wtime();
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
                    t4 = MPI_Wtime();
                    comp_time += (t4 - t3);
                } else if(rank == i+1){
                    // MPI_Recv(local_array_tmp, counts[i], MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // MPI_Send(local_array, counts[i+1], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    tc1 = MPI_Wtime();
                    MPI_Sendrecv(local_array, counts[i+1], MPI_FLOAT, i, 0,
                    local_array_tmp, counts[i], MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    tc2 = MPI_Wtime();
                    comm_time += (tc2 - tc1);

                    t3 = MPI_Wtime();
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
                    t4 = MPI_Wtime();
                    comp_time += (t4 - t3);
                }
            }
        }
        tc1 = MPI_Wtime();
        MPI_Allreduce(&local_sorted, &sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        tc2 = MPI_Wtime();
        comm_time += (tc2 - tc1);
    }
    
    t1 = MPI_Wtime();
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
    t2 = MPI_Wtime();
    io_time += (t2 - t1);

    // tc1 = MPI_Wtime();
    // MPI_Gatherv(local_array, counts[rank], MPI_FLOAT,
    //             global_array, counts, displs, MPI_FLOAT,
    //             0, MPI_COMM_WORLD);
    // tc2 = MPI_Wtime();
    // comm_time += (tc2 - tc1);

    // if (rank == 0) {
    //     t1 = MPI_Wtime();
    //     MPI_File_open(MPI_COMM_SELF, output_filename,
    //                   MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    //     MPI_File_write_at(output_file, 0, global_array, N, MPI_FLOAT, MPI_STATUS_IGNORE);
    //     MPI_File_close(&output_file);
    //     free(global_array);
    //     t2 = MPI_Wtime();
    //     io_time += (t2 - t1);
    // }
    end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    double sum_io, sum_comm, sum_comp, sum_total;
    // MPI_Reduce(&io_time,   &sum_io,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&comm_time, &sum_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&comp_time, &sum_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&total_time,&sum_total,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // // rank 0 prints the maximum times
    // if (rank == 0) {
    //     printf("Total max times across all ranks:\n");
    //     printf(" IO Time   = %.6f sec\n", sum_io);
    //     printf(" Comm Time = %.6f sec\n", sum_comm);
    //     printf(" Comp Time = %.6f sec\n", sum_comp);
    //     printf(" Total     = %.6f sec\n", sum_total);
    // }
    // printf("Rank %d: %f %f %f %f\n", rank, io_time, comm_time, comp_time, total_time);
    // 把每個 rank 的時間加總
    MPI_Reduce(&io_time,   &sum_io,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &sum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &sum_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time,&sum_total,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // rank 0 印出總和
    if (rank == 0) {
        printf("Total summed times across all ranks:\n");
        printf("  IO Time   = %.6f sec\n", sum_io/active_procs);
        printf("  Comm Time = %.6f sec\n", sum_comm/active_procs);
        printf("  Comp Time = %.6f sec\n", sum_comp/active_procs);
        printf("  Total     = %.6f sec\n", sum_total/active_procs);
    }
    free(local_array);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

