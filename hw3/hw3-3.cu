#include <stdio.h>
#include <omp.h>
#include <time.h>

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
    printf("V: %d, E: %d\n", V, E);
    pad_V = (V%block_size==0)?V:(V/block_size+1)*block_size;
    matrix = (int*)malloc(sizeof(int)*pad_V*pad_V);
    for(int i=0;i<pad_V;i++){
        for(int j=0;j<pad_V;j++){
            if(i==j){
                matrix[i*pad_V+j]=0;
            }
            else {
                matrix[i*pad_V+j]=INF;
            }
        }
    }
    for(int i=0;i<E;i++){
        int tmp[3];
        fread(tmp, sizeof(int), 3, file);
        matrix[tmp[0]*pad_V+tmp[1]]=tmp[2];
    }
    fclose(file);
}

void output(char* output_file){
    FILE* file = fopen(output_file, "w");
    for(int i=0;i<V;i++){
        fwrite(matrix+(i*pad_V), sizeof(int),  V, file);
    }
	fclose(file);
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

__global__ void floyd_phase3(int* device_matrix, int pad_V, int round, int yOffset){
    if(blockIdx.x==round||blockIdx.y+yOffset==round){
        return;
    }
    /*calculate x, y*/
    int x = threadIdx.x;
    int y = threadIdx.y;
    int start_x = blockIdx.x*block_size;
    int start_y = (blockIdx.y + yOffset)*block_size;

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
    read_input(input_file);
    int* device_matrix[2];
    //pin
    cudaHostRegister(matrix, sizeof(int)*pad_V*pad_V, cudaHostRegisterDefault);
    // cudaMalloc(&device_matrix, sizeof(int)*pad_V*pad_V);
    // cudaMemcpyAsync(device_matrix, matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyHostToDevice);

    int num_block = pad_V/block_size;
    dim3 phase1(1,1);
    dim3 phase2(num_block, 2);
    // dim3 blockPerGrid(num_block, num_block);
    dim3 threadPerBlock(thread_size, thread_size);
    struct timespec start, end;
    double cpu_time_used;
    
    #pragma omp parallel num_threads(2)
    {
        int threadId = omp_get_thread_num();
		cudaSetDevice(threadId);
		cudaMalloc((void**)&device_matrix[threadId], pad_V * pad_V * sizeof(int));

        dim3 gridPhase3(num_block, num_block / 2);
		if (threadId == 1 && (num_block & 1)) ++gridPhase3.y;   //handle odd num_block

        int yOffset = (threadId == 0) ? 0 : num_block / 2;

         int total_round = num_block;
        cudaMemcpy(device_matrix[threadId] + yOffset * block_size * pad_V, matrix + yOffset * block_size * pad_V, gridPhase3.y * block_size * pad_V * sizeof(int), cudaMemcpyHostToDevice);
        // struct timespec start, end, temp;
        // double time_used;   
        // clock_gettime(CLOCK_MONOTONIC, &start); 
        for (int round = 0; round < total_round; ++round) {
			if (round >= yOffset && round < yOffset + gridPhase3.y) {
				cudaMemcpy(matrix + round * block_size * pad_V, device_matrix[threadId] + round * block_size * pad_V, block_size * pad_V * sizeof(int), cudaMemcpyDeviceToHost);
			}
			#pragma omp barrier

			if (round < yOffset || round >= yOffset + gridPhase3.y) {
				cudaMemcpy(device_matrix[threadId] + round * block_size * pad_V, matrix + round * block_size * pad_V, block_size * pad_V * sizeof(int), cudaMemcpyHostToDevice);
			}

			floyd_phase1<<<phase1, threadPerBlock>>>(device_matrix[threadId], pad_V, round);
            floyd_phase2<<<phase2, threadPerBlock>>>(device_matrix[threadId], pad_V, round, num_block);
            floyd_phase3<<<gridPhase3, threadPerBlock>>>(device_matrix[threadId], pad_V, round, yOffset);
        }
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // if ((end.tv_nsec - start.tv_nsec) < 0) {
        //     temp.tv_sec = end.tv_sec-start.tv_sec-1;
        //     temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        // } else {
        //     temp.tv_sec = end.tv_sec - start.tv_sec;
        //     temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        // }
        // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;

        // printf("%f second\n", time_used); 
        cudaMemcpy(matrix + yOffset * block_size * pad_V, device_matrix[threadId] + yOffset * block_size * pad_V, gridPhase3.y * block_size * pad_V * sizeof(int), cudaMemcpyDeviceToHost);
    }
   
    // for(int round=0;round<total_round;round++){
    //     floyd_phase1<<<phase1, threadPerBlock>>>(device_matrix, pad_V, round);
    //     floyd_phase2<<<phase2, threadPerBlock>>>(device_matrix, pad_V, round, num_block);
    //     floyd_phase3<<<blockPerGrid, threadPerBlock>>>(device_matrix, pad_V, round);
    // }
    // cudaMemcpyAsync(matrix, device_matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyDeviceToHost);
    // cudaFree(device_matrix);
    output(output_file);
    // free(matrix);  
}