---
title: PP2024 HW3

---

# PP2024 HW3
113062574 葛奕宣

## Implementation 
1. Which algorithm do you choose in hw3-1?
Basic Floyd-warshall
3. How do you divide your data in hw3-2, hw3-3?
Since my block_size is 78 and threads per block is (26,26). So one thread is responsible for 9 data, each data in distance of 1/3*block_size.
Eg. Thread(0,0) is responsible for: (0,0), (0,1/3*blocksize), (0, 2*1/3*blocksize), (1/3*blocksize, 0), (1/3*blocksize, 1/3*blocksize), (1/3*blocksize, 2*1/3*blocksize)…… total 9 data.
In 3-3, I let each GPU do half of the block in phase 3, upper half and lower half
5. What’s your configuration in hw3-2, hw3-3? And why? (e.g. blocking factor, #blocks, #threads)
I choose B(block_size) as 78, threads per block as (26*26). #block is V after padding divided by block_size. The reason for 78 is to utilize the space of share memory.
Total shared memory is 49152, we need 2 matrix in phase 2, 3. So, it is sqrt(49152/4/2). Max block_size is 78. And the thread size limit is 1024. So I choose 26, 1/3 of 78. I’ve tested (78,15) thread size, where each thread is responsible for only 6 data. But the result is worse
7. How do you implement the communication in hw3-3?
Because each gpu do half of the blocks in phase 3. I need to copy the calculated result after half of the total round is finished. Otherwise, the calculation in phase 1 will be wrong since the data isn’t the newest. So, I copy the data back to host and copy the data back to device. The ideal method is to copy from device to device. But I got a worse result
9. Briefly describe your implementations in diagrams, figures or sentences.
* 3-1: OpenMP Parallelization
    - **Approach**: The normal Floyd-Warshall algorithm is implemented.
    - **Parallelization**: The second loop (each `i`) is parallelized using OpenMP.
    - **Objective**: Speed up the computation by dividing the workload across multiple threads in the OpenMP framework.

---

* 3-2: GPU Implementation with Block-Based Calculation
    1. **Padding**: The input graph `V` is padded to align with `block_size`.
    #### Phase 1:
    - **Blocks**: 
      - A single block is used to calculate the result.
    - **Threads**:
      - Inside the block, each thread is responsible for 9 data points.

    #### Phase 2:
    - **Blocks**: 
      - `(2 * V/blocksize)` blocks are used for computation.
    - **Responsibility**:
      - Each block calculates one block of data, excluding the block processed in Phase 1.

    #### Phase 3:
    - **Blocks**: 
      - `(V/blocksize, V/blocksize)` blocks are used to calculate all remaining blocks.
    - **Exclusion**:
      - Excludes blocks calculated in Phases 1 and 2.

---

* 3-3: Multi-GPU Optimization
    - **Difference from 3-2**: Phase 3 is optimized using two GPUs.
    - **Work Distribution**:
      - **GPU 1**: Processes the upper half of the data.
      - **GPU 2**: Processes the lower half of the data.

---

### Profiling Results (hw3-2)
I use p12k1 as testcase, below table gives the average value.
* Sample command
```bash=
srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof --kernels floyd_phase3 --metrics sm_efficiency ./hw3-2 /home/pp24/share/hw3-2/testcases/p12k1 p12k1.out
```

| Metric                          | Avg Value          |
|---------------------------------|----------------|
| **Occupancy**                   | 0.651183       |
| **SM Efficiency**               | 99.93%         |
| **Shared Memory Load Throughput** | 2575.4 GB/s    |
| **Shared Memory Store Throughput** | 258.14 GB/s    |
| **Global Load Throughput**      | 48.103 GB/s    |
| **Global Store Throughput**     | 114.47 GB/s    |

### Experiment & Analysis

#### Blocking Factor (hw3-2)
I use c05.1 as testcases.
Bandwidth is calculated by adding load and store throughput.
GOps is calculated by integer instruction / execution time

![image](https://i.imgur.com/sfaIMpQ.png)

For GOPs, when blocking factor increase, the number of threads increase, so the instruction done
per time is positive correlated to them. However, we can see 60,20 performs better than 72,24. This
may result in different padding result according to the testcases.

![image](https://i.imgur.com/qxvyqzY.png)
We can see by bandwidth experiment. When Blocking factor is large, the more we utilize the share
memory, so the bandwidth raises. Since the use of global memory doesn’t change in my program,
the bandwidth remains the same.

### Optimization (hw3-2)
1. Utilize the share memory space. Experiment is similar to the blocking factor above.
2. Use unroll to reduce branching and change if else statement to bitwise operation. There’s no
branching in my kernels.
![image](https://i.imgur.com/9bmjuIG.png)
3. Use memcpyAsync. Overlaps the copy time.
4. I tried the pitch and 2Dmemcpy, but the result is worse.
5. Cut down the use of metrics in share memory. In phase 3, I use 9 register instead of share memory for the local value.
![image](https://i.imgur.com/MnKtvvj.png)
6. Padding the input matrix so that it is aligned with the block_size. This is to reduce the branch condition inside kernel. Since we have enough blocks, the extra computation overhead is lower than the warp divergence resulted by branch conditions.

* I use c21.1 as testcase, since the gpu execution time is pretty low(1s), some performance optimization is not obvious. However, we can still see the improvement through each technique. 
![image alt](https://i.imgur.com/WTJPDV5.png)
### Weak scalability
I choose these two testcase since they have roughly 2 times of workload.
| TestCase         | V      | E        | #GPU|
|-------------------|--------|----------|-----|
| 3-2/p12k1        | 500    | 134307   |1|
| 3-3/c03.1        | 999    | 418232   |2|

![f](https://i.imgur.com/i7XNmfA.png)
The overall weak scalability is acceptable, since the runtime only increase about 25%. I think the major increase is the memory copy overhead in multi-gpu.
### Time distribution
I use p31k1 as testcase. Since phase3 dominate the execution time, I seperate the diagram into two.
![image](https://i.imgur.com/tGmTWKr.png)
1. **Phase 3**:
   - Dominates execution time (**83.3%**).
   - Key target for optimization to improve overall performance.

2. **Others Breakdown**:
   - **Output I/O** (**49.9%**) and **Input I/O** (**25.7%**) are the largest contributors outside computation.
   - **Memory Transfers** (H2D: **7.4%**, D2H: **6.6%**) are significant, especially in multi-GPU setups.

3. **Optimization Focus**:
   - Prioritize **Phase 3** computation.
   - Reduce **I/O overhead** (Output/Input).
   - Optimize **memory transfer strategies** for better scalability.

### Experiment on AMD GPU
#### Codes
the only difference is the prefix of Cuda Api are changed to hip
#### Performance in single and multi-gpu
Interestingly, in both single and multi-gpu, when the workload size is relatively small(before c20.1). RTX1080 out performs MI210. However, after that MI210 shows significant reduction in execution time. I.E, testcase p34k1, MI210 is twice faster than rtx1080.
#### Insight
In my opinion, the above result shows that HIP api may have more overhead  than  CUDA api since MI210 is slower in low computation case.
MI210 is newer than rtx1080, its computation power excels 1080. We can see this in large case obviously. 
However, when using AMD gpu, it's crucial to consider api overhead in lower cases.

### Experience & conclusion
This was a great assignment, and I learned a lot about programming with CUDA. There are many intricate details involved, and I also had the opportunity to work with AMD GPUs. Profiling their behavior and analyzing statistics was a valuable experience. Tools like nvprof are crucial and will be important for my future development in this field.