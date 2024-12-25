#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <algorithm>
#include <string>
#include <nvtx3/nvToolsExt.h>

void merge_array_left(float* &my, float* received, float* &ans, int my_size, int other_size){
    int l1, r1 ,ans_num;
    r1 = ans_num = my_size-1;
    l1 = other_size-1;

    while(ans_num>=0 ){
        if(my[r1]<=received[l1]||r1<0){
            ans[ans_num--] = received[l1--];
        }
        else {
            ans[ans_num--] = my[r1--];
        }
    }
    std::swap(my, ans);
}

void merge_array_right(float* &my, float* received, float* &ans, int my_size, int other_size){
    int l1, r1 ,ans_num;
    l1 = r1 = ans_num = 0;

    while(ans_num<my_size ){
        if(my[l1]<=received[r1]||r1>=other_size) {
            ans[ans_num++] = my[l1++];
        }
        else {
            ans[ans_num++] = received[r1++];
        }
    }
    std::swap(my, ans);
}

int main(int argc, char **argv) {

    nvtxRangePush("main");


    MPI_Init(&argc, &argv);

    // double IO_time=0.0, Com_time=0.0, start_time=0.0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_File input_file, output_file;

    int array_size = std::stoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    // Split the communicator
    int num_used_procs = std::min(array_size, world_size);
    MPI_Comm new_comm;
    int color = (rank < num_used_procs) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);

    // when array is small
    const int threshold = 1024783; 
    if (array_size <= threshold) {
        // Only rank 0 will handle the sorting and writing
        if (rank == 0) {
            float *my_array = new float[array_size];

            MPI_File_open(MPI_COMM_SELF, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
            MPI_File_read_at(input_file, 0, my_array, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);

            // Sort the array using spreadsort
            boost::sort::spreadsort::float_sort(my_array, my_array + array_size);

            MPI_File_open(MPI_COMM_SELF, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
            MPI_File_write_at(output_file, 0, my_array, array_size, MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        return 0;
    }

    int rank_size, rest_size;
    int left_neighbor, right_neighbor;
    int left_count, right_count;
    int start;

    rank_size = array_size/world_size;
    rest_size = array_size%world_size;
    if(rank<rest_size) rank_size++;

    if(rank==0) left_neighbor = -1;
    else left_neighbor = rank-1;
    // 2nd case handle when P>N
    if(rank==world_size-1 || (rank==rest_size-1&&rank_size==1)) right_neighbor = -1;
    else right_neighbor = rank+1;

    //start
    if(rank<rest_size){
        start = rank_size*rank;
    }
    else {
        start = (rank_size+1)*rest_size + (rank-rest_size)*rank_size;
    }

    //left_count, right_count
    if(rank==rest_size-1){
        left_count = rank_size;
        right_count = rank_size+1;
    }
    else if(rank==rest_size){
        left_count = rank_size+1;
        right_count = rank_size;
    }
    else{
        left_count = right_count = rank_size;
    }

    if(rank_size == 0){
        left_neighbor = right_neighbor = left_count = right_count = -1;
    }
    float *my_array = new float[rank_size];
    float *received_array = new float[rank_size+1];
    float *result_array = new float[rank_size];
    
    nvtxRangePush("IO");
    MPI_File_open(new_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if(rank_size!=0) {
        MPI_File_read_at(input_file, sizeof(float) *start, my_array, rank_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    nvtxRangePop();
    // MPI_File_close(&input_file);
    //sort local
    if(rank_size!=0){
        boost::sort::spreadsort::float_sort(my_array, my_array+rank_size);
    }
    // int t = 2;
    float received_num;
    // if(world_size>=5){
    //     while(t--){
    //         //odd round
    //         if(rank&1){
    //             if(right_neighbor!=-1){
    //                 // start_time = MPI_Wtime();
    //                 MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                 // Com_time+= MPI_Wtime() - start_time;
    //                 if(received_num<my_array[rank_size-1]){
    //                     // start_time = MPI_Wtime();
    //                     MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                     // Com_time+= MPI_Wtime() - start_time;
    //                     merge_array_right(my_array, received_array, result_array, rank_size, right_count);
    //                 }
    //             }
    //         }
    //         else {
    //             if(left_neighbor!=-1){
    //                 MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                 if(received_num>my_array[0]){
    //                     MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                     merge_array_left(my_array, received_array, result_array, rank_size, left_count);
    //                 }
    //             }
    //         }
    //         //even round
    //         if(rank&1){
    //             if(left_neighbor!=-1){
    //                 MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                 if(received_num>my_array[0]){
    //                     MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                     merge_array_left(my_array, received_array, result_array, rank_size, left_count);
    //                 }
    //             }
    //         }
    //         else {
    //             if(right_neighbor!=-1){
    //                 MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                 if(received_num<my_array[rank_size-1]){
    //                     MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
    //                     merge_array_right(my_array, received_array, result_array, rank_size, right_count);
    //                 }
    //             }
    //         }
    //     }
    // }
    
    int all_sort=0;
    int sort = 0;
    while(all_sort < world_size){
        nvtxRangePush("Loop");
        sort = 1;
        //odd round
        if(rank&1){
            if(right_neighbor!=-1){
                MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                if(received_num<my_array[rank_size-1]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                    merge_array_right(my_array, received_array, result_array, rank_size, right_count);
                }
            }
        }
        else {
            if(left_neighbor!=-1){
                MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                if(received_num>my_array[0]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                    merge_array_left(my_array, received_array, result_array, rank_size, left_count);
                }
            }
        }
        //even round
        if(rank&1){
            if(left_neighbor!=-1){
                MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                if(received_num>my_array[0]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                    merge_array_left(my_array, received_array, result_array, rank_size, left_count);
                }
            }
        }
        else {
            if(right_neighbor!=-1){
                MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                if(received_num<my_array[rank_size-1]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, new_comm, MPI_STATUS_IGNORE);
                    merge_array_right(my_array, received_array, result_array, rank_size, right_count);
                }
            }
        }
        nvtxRangePop();
        MPI_Allreduce(&sort, &all_sort, 1, MPI_INT, MPI_SUM, new_comm);
    }
    
    nvtxRangePush("IO");
    MPI_File_open(new_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (rank_size != 0) {
        MPI_File_write_at(output_file, sizeof(float) * start, my_array, rank_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    nvtxRangePop();
    // MPI_File_close(&output_file);
    // delete []my_array;
    // delete []received_array;
    // delete []result_array;

    // MPI_Finalize();
    nvtxRangePop();
    return 0;
}


