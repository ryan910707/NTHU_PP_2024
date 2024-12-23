#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

__device__ float _max(float a, float b) { return a > b ? a : b; }
__device__ float _min(float a, float b) { return a < b ? a : b; }

// Global variables as before
int B, N, d;
float *Q, *K, *V, *O;

// Adjust block tiling
#define br 32
#define bc 32
#define d_offset 65  // as in your code

// Host helpers
void input(char *input_filename);
void output(char *output_filename);

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
}

/**
 * @brief Single kernel that processes all B batches.
 *
 * Each block (blockIdx.x, blockIdx.y) corresponds to:
 *  - blockIdx.y = b in [0..B-1] (batch index)
 *  - blockIdx.x = tile of rows in [0..(N/br)-1].
 *
 * Each thread block is (br=32, bc=32).
 *  - threadIdx.y in [0..31] => sub-row within the tile
 *  - threadIdx.x in [0..31] => sub-col
 *
 * We keep the "for (int j=0; j<tc; j++)" inside the kernel
 * for chunking over columns in K & V (i.e. N is split in bc=32 column chunks).
 */
__global__ void flash_attention(
    float *q, float *k, float *v, float *o, float *l,
    int d, int N, int B
) {
    // figure out which row we are processing
    int b   = blockIdx.y;                // which batch
    int row = blockIdx.x * br + threadIdx.y;

    // how many column tiles?
    int tc = N / bc; // assume N is multiple of 32

    // Pointers offset for the current batch
    // Q, K, V, O are shaped [B, N, d], so:
    float *Qb = q + (size_t)b * N * d;
    float *Kb = k + (size_t)b * N * d;
    float *Vb = v + (size_t)b * N * d;
    float *Ob = o + (size_t)b * N * d;
    float *Lb = l + (size_t)b * N;

    // Shared memory allocations
    __shared__ float kj[bc * d_offset];
    __shared__ float vj[bc * d_offset];
    __shared__ float sij[br * bc];   // S = QK^T * scalar
    __shared__ float pij[br * bc];   // P = softmax(S)
    __shared__ float qi[br * d_offset];
    __shared__ float oi[br * d_offset];
    __shared__ float li[br];
    __shared__ float mij[br];  // row max
    __shared__ float lij[br];  // row sum

    // 1) Load Q and O for this row from global memory
    //    We chunk the dimension d in steps of bc=32
    int chunkCount = d / bc; // assume d multiple of 32
    #pragma unroll 32
    for (int r = 0; r < chunkCount; r++) {
        // Q[row, r*bc + threadIdx.x]
        qi[threadIdx.y * d_offset + threadIdx.x + r * bc] =
            Qb[row * d + (r * bc + threadIdx.x)];

        // O[row, r*bc + threadIdx.x]  (initial partial)
        oi[threadIdx.y * d_offset + threadIdx.x + r * bc] =
            Ob[row * d + (r * bc + threadIdx.x)];
    }
    // Load partial l
    if (threadIdx.x == 0) {
        li[threadIdx.y] = Lb[row];
    }
    __syncthreads();

    // 2) For each column tile j in [0..tc-1]
    #pragma unroll 32
    for (int j = 0; j < tc; j++) {
        // 2.a) Load K and V for these bc columns into shared memory
        // The row in shared memory is threadIdx.y, the col is threadIdx.x,
        // but here we actually store in [threadIdx.y*d_offset + threadIdx.x].
        #pragma unroll 32
        for (int i = 0; i < chunkCount; i++) {
            // Kb[ (j*bc + threadIdx.y)*d + (i*bc + threadIdx.x) ]
            kj[threadIdx.y * d_offset + threadIdx.x + i * bc] =
                Kb[(size_t)(j * bc + threadIdx.y) * d + (i * bc + threadIdx.x)];
            vj[threadIdx.y * d_offset + threadIdx.x + i * bc] =
                Vb[(size_t)(j * bc + threadIdx.y) * d + (i * bc + threadIdx.x)];
        }
        __syncthreads();

        // 2.b) QK^T dot product for (row, j*bc..(j+1)*bc)
        {
            int col = threadIdx.x;
            float val = 0.0f;
            // compute dot for dimension d
            #pragma unroll 32
            for (int t = 0; t < d; t++) {
                val += qi[threadIdx.y * d_offset + t]
                      * kj[col * d_offset + t];
            }
            // scale by 1 / sqrt(d)
            sij[threadIdx.y * bc + col] = val * (1.0f / sqrtf((float)d));
        }
        __syncthreads();

        // 2.c) RowMax (sequential or warp-based).
        // Here, do a simple sequential approach for clarity.
        {
            if (threadIdx.x == 0) {
                float max_val = sij[threadIdx.y * bc];
                for (int i = 1; i < bc; i++) {
                    float tmp = sij[threadIdx.y * bc + i];
                    if (tmp > max_val) max_val = tmp;
                }
                mij[threadIdx.y] = max_val;
            }
        }
        __syncthreads();

        // 2.d) Subtract max and exponentiate
        {
            int col = threadIdx.x;
            float m = mij[threadIdx.y];
            float val = sij[threadIdx.y * bc + col] - m;
            pij[threadIdx.y * bc + col] = __expf(val);
        }
        __syncthreads();

        // 2.e) RowSum
        {
            if (threadIdx.x == 0) {
                float sum_val = 0.0f;
                #pragma unroll 32
                for (int i = 0; i < bc; i++) {
                    sum_val += pij[threadIdx.y * bc + i];
                }
                lij[threadIdx.y] = sum_val;
            }
        }
        __syncthreads();

        // 2.f) Update partial O and partial l
        {
            __shared__ float li_new[br];
            int i = threadIdx.y;
            int col = threadIdx.x;

            if (col == 0) {
                // li_new = li + e^m_ij * row_sum
                li_new[i] = li[i] + __expf(mij[i]) * lij[i];
            }
            __syncthreads();

            // Update O
            #pragma unroll 32
            for (int r = 0; r < chunkCount; r++) {
                float pv = 0.0F;
                // sum over these bc columns
                #pragma unroll 32
                for (int t = 0; t < bc; t++) {
                    pv += pij[i * bc + t] *
                          vj[t * d_offset + (col + r * bc)];
                }
                // Weighted combination
                // O_i = ( li[i]*old_Oi + e^mij[i]*pv ) / li_new[i]
                float oldVal = oi[i * d_offset + (col + r * bc)];
                float numerator = li[i] * oldVal + __expf(mij[i]) * pv;
                oi[i * d_offset + (col + r * bc)] = numerator / li_new[i];
            }
            __syncthreads();

            // update li
            if (col == 0) {
                li[i] = li_new[i];
            }
        }
        __syncthreads();
    } // end for (j in [0..tc-1])

    // 3) Write final O and l back to global memory
    #pragma unroll 32
    for (int r = 0; r < chunkCount; r++) {
        Ob[row * d + (r * bc + threadIdx.x)] =
            oi[threadIdx.y * d_offset + (r * bc + threadIdx.x)];
    }
    if (threadIdx.x == 0) {
        Lb[row] = li[threadIdx.y];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    // Read input
    input(argv[1]);

    // Allocate device memory
    float *device_Q, *device_K, *device_V, *device_O, *device_l;
    cudaMalloc(&device_Q, sizeof(float)*N*B*d);
    cudaMalloc(&device_K, sizeof(float)*N*B*d);
    cudaMalloc(&device_V, sizeof(float)*N*B*d);
    cudaMalloc(&device_O, sizeof(float)*N*B*d);
    cudaMalloc(&device_l, sizeof(float)*N*B);

    // Optional: pinned host registration, asynchronous copies
    cudaHostRegister(Q, sizeof(float)*N*B*d, cudaHostRegisterDefault);
    cudaHostRegister(K, sizeof(float)*N*B*d, cudaHostRegisterDefault);
    cudaHostRegister(V, sizeof(float)*N*B*d, cudaHostRegisterDefault);

    // Copy inputs to device
    cudaMemcpyAsync(device_Q, Q, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(device_K, K, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(device_V, V, sizeof(float)*N*B*d, cudaMemcpyHostToDevice);
    cudaMemset(device_O, 0, sizeof(float)*N*B*d);
    cudaMemset(device_l, 0, sizeof(float)*N*B);

    // Launch the kernel ONCE
    int tr = N / br;  // how many row-tiles
    dim3 blockPerGrid(tr, B);
    dim3 threadPerBlock(br, bc); // (32, 32)

    flash_attention<<<blockPerGrid, threadPerBlock>>>(
        device_Q, device_K, device_V, device_O, device_l,
        d, N, B
    );

    // Copy result back
    cudaMemcpyAsync(O, device_O, sizeof(float)*N*B*d, cudaMemcpyDeviceToHost);

    // Output
    FILE *file = fopen(argv[2], "wb");
    fwrite(O, sizeof(float), B * N * d, file);

    return 0;
}
