#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);

__device__ float _max(float a, float b) { return a > b ? a : b; }
__device__ float _min(float a, float b) { return a < b ? a : b; }

int B, N, d;
float *Q, *K, *V, *O;
#define br 32
#define bc 32
#define d_offset 64

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

__device__ void QKDotAndScalar(float *out, float *q, float *k, float scalar, int d) {
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    float val = 0.0f;
    for (int t = 0; t < d; t++) {
        val += q[row * d_offset + t] * k[col * d_offset + t];
    }
    out[row * bc + col] = val * scalar;
}

__device__ void RowMax(float *out, float *in) {
    int row = threadIdx.y;
    // Only one thread per row computes the maximum
    if (threadIdx.x == 0) {
        float max_val = in[row * bc];
        for (int i = 0; i < bc; i++) {
            max_val = _max(max_val, in[row * bc + i]);
        }
        out[row] = max_val;
    }
}

__device__ void MinusMaxAndExp(float *out, float *in, float *mx) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    out[row * bc + col] = expf(in[row * bc + col] - mx[row]);
}

__device__ void RowSum(float *out, float *in) {
    int row = threadIdx.y;
    // Only one thread per row computes the sum
    if (threadIdx.x == 0) {
        float sum_val = 0.0f;
        for (int i = 0; i < bc; i++) {
            sum_val += in[row * bc + i];
        }
        out[row] = sum_val;
    }
}

__device__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int d) {
    __shared__ float mi_new[br];
    __shared__ float li_new[br];
    int i = threadIdx.y;
    int j = threadIdx.x;

    if(threadIdx.x == 0){ 
        mi_new[i] = _max(mi[i], mij[i]);
        li_new[i] = expf(mi[i] - mi_new[i]) * li[i] + expf(mij[i] - mi_new[i]) * lij[i];
    }
    __syncthreads();

    for(int round =0; round <d/bc; round++){
        float pv = 0.0F;
        for (int t = 0; t < bc; t++) {
            pv += pij[i * bc + t] * vj[t * d_offset + j+round*bc];
        }
        oi[i * d_offset + j+round*bc] = (li[i] * expf(mi[i] - mi_new[i]) * oi[i * d_offset + j+round*bc] + expf(mij[i] - mi_new[i]) * pv) / li_new[i];
    }
    __syncthreads();

    if(threadIdx.x == 0){
        mi[i] = mi_new[i];
        li[i] = li_new[i];
    }   
}

__global__ void flash_attention(float *q, float *k, float *v, float *o, float* l, float* m, int d, int tc, int N){
    for (int j = 0; j < tc; j++){
        //load k and v to shared memory
        __shared__ float kj[bc*d_offset];
        __shared__ float vj[bc*d_offset];
        int round = d/bc;
        for(int i=0;i<round;i++){
            kj[threadIdx.y*d_offset+threadIdx.x+i*bc] = k[j*bc*d+threadIdx.y*d+threadIdx.x+i*bc];
            vj[threadIdx.y*d_offset+threadIdx.x+i*bc] = v[j*bc*d+threadIdx.y*d+threadIdx.x+i*bc];
        }
        __syncthreads();


        //load shared memory
        __shared__ float sij[br*bc];
        __shared__ float pij[br*bc];
        __shared__ float qi[br*d_offset];
        __shared__ float oi[br*d_offset];
        __shared__ float li[br];
        __shared__ float mi[br];
        __shared__ float mij[br];
        __shared__ float lij[br];
        
        for(int round=0; round<d/bc; round++){
            qi[threadIdx.y*d_offset+threadIdx.x+round*bc] = q[blockIdx.x*br*d+threadIdx.y*d+threadIdx.x+round*bc];
            oi[threadIdx.y*d_offset+threadIdx.x+round*bc] = o[blockIdx.x*br*d+threadIdx.y*d+threadIdx.x+round*bc];
        }

        if(threadIdx.x == 0){
            li[threadIdx.y] = l[blockIdx.x*br+threadIdx.y];
            mi[threadIdx.y] = m[blockIdx.x*br+threadIdx.y];
        }
        __syncthreads();


        QKDotAndScalar(sij, qi, kj, 1.0 / sqrtf((float)d), d);
        __syncthreads();
        RowMax(mij, sij);
        __syncthreads();
        MinusMaxAndExp(pij, sij, mij);
        __syncthreads();
        RowSum(lij, pij);
        __syncthreads();

        UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, d);

        //write back
        for(int round=0; round<d/bc; round++){
            o[blockIdx.x*br*d+threadIdx.y*d+threadIdx.x+round*bc] = oi[threadIdx.y*d_offset+threadIdx.x+round*bc];
        }
        if(threadIdx.x == 0){
            l[blockIdx.x*br+threadIdx.y] = li[threadIdx.y];
            m[blockIdx.x*br+threadIdx.y] = mi[threadIdx.y];
        }
        
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    float *device_Q;
    float *device_K;
    float *device_V;
    float *device_O;
    float *device_l;
    float *device_m;

    //copy data to device
    cudaHostRegister(Q, sizeof(float)*N*B*d, cudaHostRegisterDefault);
    cudaHostRegister(K, sizeof(float)*N*B*d, cudaHostRegisterDefault);
    cudaHostRegister(V, sizeof(float)*N*B*d, cudaHostRegisterDefault);
    cudaMalloc(&device_Q, sizeof(float)*N*B*d);
    cudaMalloc(&device_K, sizeof(float)*N*B*d);
    cudaMalloc(&device_V, sizeof(float)*N*B*d);
    cudaMalloc(&device_O, sizeof(float)*N*B*d);
    cudaMalloc(&device_l, sizeof(float)*N*B);
    cudaMalloc(&device_m, sizeof(float)*N*B);
    cudaMemcpy(device_Q, Q, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);
    cudaMemcpy(device_K, K, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, V, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);

    cudaMemset(device_O, 0, sizeof(float)*N*B*d);
    cudaMemset(device_l, 0, sizeof(float)*N*B);
    cudaMemset(device_m, FLT_MIN, sizeof(float)*N*B);

    int tr = N / br, tc = N / bc;
    dim3 blockPerGrid(tr);
    dim3 threadPerBlock(br, bc);

    for (int i = 0; i < B; i++) {
        flash_attention<<<blockPerGrid, threadPerBlock>>>(
            device_Q + (i * N * d), 
            device_K + (i * N * d), 
            device_V + (i * N * d), 
            device_O + (i * N * d),
            device_l + (i * N),
            device_m + (i * N),
            d, tc, N
        );
    }

    //copy data back to host
    cudaMemcpy(O, device_O, sizeof(float)*N*B*d, cudaMemcpyDeviceToHost);
    
    output(argv[2]);

    return 0;
}