#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#define INF 1073741823
#define block_size  78
#define thread_size  26
#define block_size_index 26

int V,E, pad_V;
int *matrix;



void read_input(char* input_file){
    FILE* file = fopen(input_file, "rb");

    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    pad_V = (V % block_size == 0) ? V : (V / block_size + 1) * block_size;

    size_t matrix_size = sizeof(int) * pad_V * pad_V;
    matrix = (int*)malloc(matrix_size);

    // Check if INF can be set using memset
    if (sizeof(int) == 4 && INF == -1) {
        memset(matrix, 0xFF, matrix_size);
    } else {
        // Parallel initialization of the matrix to INF
        #pragma omp parallel for
        for (int i = 0; i < pad_V * pad_V; i++) {
            matrix[i] = INF;
        }
    }

    // Set diagonal elements to zero
    #pragma omp parallel for
    for (int i = 0; i < pad_V; i++) {
        matrix[i * pad_V + i] = 0;
    }

    // Read all edge data at once
    int* edge_data = (int*)malloc(sizeof(int) * 3 * E);
    fread(edge_data, sizeof(int), 3 * E, file);

    // Update matrix with edge weights
    #pragma omp parallel for
    for (int i = 0; i < E; i++) {
        int src = edge_data[3 * i];
        int dst = edge_data[3 * i + 1];
        int weight = edge_data[3 * i + 2];
        matrix[src * pad_V + dst] = weight;
    }
}
__global__ void floyd_phase1(int* device_matrix, int pad_V, int round){
    /*calculate x, y*/
    int x = threadIdx.x;  //0~23
    int y = threadIdx.y;

    /*copy device matrix needed to cache*/
    __shared__ int share_matrix[block_size][block_size];
    
    #pragma unroll 
    for(int i=0;i<3;i++){
        #pragma unroll
        for(int j=0;j<3;j++){
            share_matrix[y+i*block_size_index][x+j*block_size_index] = device_matrix[(round*block_size+y+i*block_size_index)*pad_V+(round*block_size+x+j*block_size_index)];
        }
    }
    __syncthreads();

    /*calculation*/
    #pragma unroll 
    for(int k=0;k<block_size;k++){
        #pragma unroll
        for(int i=0;i<3;i++){
            #pragma unroll
            for(int j=0;j<3;j++){
                share_matrix[y+i*block_size_index][x+j*block_size_index] = min(share_matrix[y+i*block_size_index][x+j*block_size_index], share_matrix[y+i*block_size_index][k]+share_matrix[k][x+j*block_size_index]);
            }
        }
        __syncthreads();
    }

    /*write back*/
    #pragma unroll
    for(int i=0;i<3;i++){
        #pragma unroll
        for(int j=0;j<3;j++){
            device_matrix[(round*block_size+y+i*block_size_index)*pad_V+(round*block_size+x+j*block_size_index)] = share_matrix[y+i*block_size_index][x+j*block_size_index];
        }
    }
}

__global__ void floyd_phase2(int* device_matrix, int pad_V, int round, int num_block){
    if(blockIdx.x==round){
        return;
    }
    /*calculate x, y*/
    int x = threadIdx.x;  
    int y = threadIdx.y;
    int start_x, start_y;

    start_x = blockIdx.x*block_size*blockIdx.y + round*block_size*(!blockIdx.y);
    start_y = round*block_size*blockIdx.y + blockIdx.x*block_size*(!blockIdx.y);

    /*copy device matrix needed to cache*/
    __shared__ int share_my[block_size][block_size];
    __shared__ int share_phase1[block_size][block_size];

    #pragma unroll 3
    for(int i=0;i<3;i++){
        #pragma unroll 3
        for(int j=0;j<3;j++){
            share_my[3*y+i][3*x+j] = device_matrix[(start_y+3*y+i)*pad_V+(start_x+3*x+j)];
            share_phase1[3*y+i][3*x+j] = device_matrix[(round*block_size+3*y+i)*pad_V+(round*block_size+3*x+j)];
        }
    }
    __syncthreads();


    /*calculation*/
    #pragma unroll 
    for(int k=0;k<block_size;k++){
        #pragma unroll 3
        for(int i=0;i<3;i++){
            #pragma unroll 3
            for(int j=0;j<3;j++){
                int idx1 = (3*y+i)*blockIdx.y+(k)*(!blockIdx.y);
                int idx2 = (3*x+j)*(!blockIdx.y)+(k)*(blockIdx.y);
                int idx3 = k*blockIdx.y+(3*y+i)*(!blockIdx.y);
                int idx4 = k*(!blockIdx.y)+(3*x+j)*blockIdx.y;
                share_my[3*y+i][3*x+j] = min(share_my[3*y+i][3*x+j], share_phase1[idx1][idx2]+share_my[idx3][idx4]);
            }
        }
    }
    
    /*write back*/
    #pragma unroll 3
    for(int i=0;i<3;i++){
        #pragma unroll 3
        for(int j=0;j<3;j++){
            device_matrix[(start_y+3*y+i)*pad_V+(start_x+3*x+j)] = share_my[3*y+i][3*x+j];
        }
    }
}

__global__ void floyd_phase3(int* device_matrix, int pad_V, int round){
    if(blockIdx.x==round||blockIdx.y==round){
        return;
    }
    /*calculate x, y*/
    int x = threadIdx.x;
    int y = threadIdx.y;
    int start_x = blockIdx.x*block_size;
    int start_y = blockIdx.y*block_size;

    int dy = start_y+3*y;
    int dx = start_x+3*x;
    int ans1 = device_matrix[(start_y+y)*pad_V+(start_x+x)];
    int ans2 = device_matrix[(start_y+y)*pad_V+(start_x+x+block_size_index)];
    int ans3 = device_matrix[(start_y+y)*pad_V+(start_x+x+block_size_index*2)];
    int ans4 = device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x)];
    int ans5 = device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x+block_size_index)];
    int ans6 = device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x+block_size_index*2)];
    int ans7 = device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x)];
    int ans8 = device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x+block_size_index)];
    int ans9 = device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x+block_size_index*2)];

    /*copy device matrix needed to cache*/
    __shared__ int share_row[block_size][block_size];
    __shared__ int share_col[block_size][block_size];
    #pragma unroll 3
    for(int i=0;i<3;i++){
        #pragma unroll 3
        for(int j=0;j<3;j++){ 
            share_row[3*y+i][3*x+j] = device_matrix[(dy+i)*pad_V+(round*block_size+3*x+j)];   //load data with same row
            share_col[3*y+i][3*x+j] = device_matrix[(round*block_size+3*y+i)*pad_V+(dx+j)];    //same col
        }
    }
    __syncthreads();

    /*calculation*/
    #pragma unroll  
    for(int k=0;k<block_size;k++){
        ans1 = min(ans1, share_row[y][k]+share_col[k][x]);
        ans2 = min(ans2, share_row[y][k]+share_col[k][x+block_size_index]);
        ans3 = min(ans3, share_row[y][k]+share_col[k][x+block_size_index*2]);
        ans4 = min(ans4, share_row[y+block_size_index][k]+share_col[k][x]);
        ans5 = min(ans5, share_row[y+block_size_index][k]+share_col[k][x+block_size_index]);
        ans6 = min(ans6, share_row[y+block_size_index][k]+share_col[k][x+block_size_index*2]);
        ans7 = min(ans7, share_row[y+block_size_index*2][k]+share_col[k][x]);
        ans8 = min(ans8, share_row[y+block_size_index*2][k]+share_col[k][x+block_size_index]);
        ans9 = min(ans9, share_row[y+block_size_index*2][k]+share_col[k][x+block_size_index*2]);
    }

    /*write back*/
    device_matrix[(start_y+y)*pad_V+(start_x+x)] = ans1;
    device_matrix[(start_y+y)*pad_V+(start_x+x+block_size_index)] = ans2;
    device_matrix[(start_y+y)*pad_V+(start_x+x+block_size_index*2)] = ans3;
    device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x)] = ans4;
    device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x+block_size_index)] = ans5;
    device_matrix[(start_y+y+block_size_index)*pad_V+(start_x+x+block_size_index*2)] = ans6;
    device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x)] = ans7;
    device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x+block_size_index)] = ans8;
    device_matrix[(start_y+y+block_size_index*2)*pad_V+(start_x+x+block_size_index*2)] = ans9;
}

int main(int argc, char** argv){
    char* input_file = argv[1];
    char* output_file = argv[2];
    struct timespec start, end;
    double cpu_time_used;

    read_input(input_file);
    int* device_matrix;

    cudaHostRegister(matrix, sizeof(int)*pad_V*pad_V, cudaHostRegisterDefault);
    cudaMalloc(&device_matrix, sizeof(int)*pad_V*pad_V);
    cudaMemcpyAsync(device_matrix, matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyHostToDevice);

    int num_block = pad_V/block_size;
    dim3 phase1(1,1);
    dim3 phase2(num_block, 2);
    dim3 blockPerGrid(num_block, num_block);
    dim3 threadPerBlock(thread_size, thread_size);
    
    int total_round = num_block;
    for(int round=0;round<total_round;round++){
        floyd_phase1<<<phase1, threadPerBlock>>>(device_matrix, pad_V, round);
        floyd_phase2<<<phase2, threadPerBlock>>>(device_matrix, pad_V, round, num_block);
        // clock_gettime(CLOCK_MONOTONIC, &start);
        floyd_phase3<<<blockPerGrid, threadPerBlock>>>(device_matrix, pad_V, round);
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // double elapsed = (end.tv_sec - start.tv_sec) + 
        //              (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // printf("Time taken: %f seconds\n", elapsed);
    }
    FILE* out_file = fopen(output_file, "w");
    cudaMemcpyAsync(matrix, device_matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyDeviceToHost);
    cudaFree(device_matrix);
    for(int i=0;i<V;i++){
        fwrite(matrix+(i*pad_V), sizeof(int),  V, out_file);
    }

}