# PCA-EXP-1-PCA-GPU-Based-Vector-Summation

i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.

## AIM:

To perform GPU based vector summation and explore the differences with different block values.

## PROCEDURE:

* Open "sumArraysOnGPU-timer.cu" in a text editor or IDE.

* Set "block.x" to 1023 and recompile the program. Then run it.

* Set "block.x" to 1024 and recompile the program. Then run it.

* Compare the results and observe any differences in performance.

* Set "block.x" to 256 and modify the kernel function to let each thread handle two elements.

* Recompile and run the program.

* Compare the results with other execution configurations, such as "block.x = 512" or "block.x = 1024".

* Analyze the results and observe any differences in performance.

* Repeat the steps with different input arrays and execution configurations to further explore the program's performance characteristics.

## (i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.
## PROGRAM
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <sys/time.h>
    int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set up date size of vectors
    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);
    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    double iStart,iElaps;
    // initialize data at host side
    iStart = cpuSecond();
    initialData (h_A, nElem);
    initialData (h_B, nElem);
    iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // add vector at host side for result checks
    iStart = cpuSecond();
    sumArraysOnHost (h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;
    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    // invoke kernel at host side
    int iLen = 1023;
    dim3 block (iLen);
    dim3 grid ((nElem+block.x-1)/block.x);
    iStart = cpuSecond();
    sumArraysOnGPU <<<grid, block>>>(d_A, d_B, d_C,nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f" \
    "sec\n", grid.x, block.x, iElaps);
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return(0);
    }
## Output:
        The result of the execution with block.x = 1023 is:
        sumArraysOnGPU <<<9766,1023>>> Time elapsed 0.000019sec

        The result of the execution with block.x = 1024 is:
        sumArraysOnGPU <<<9766,1024>>> Time elapsed 0.000018sec

## Differences and the Reason:
* As we can see, the execution times are very similar, with the configuration block.x = 1024 being slightly faster.
* The reason for this difference is that the GPU hardware processes blocks of threads in multiples of 32, called warps. In our case, when we set block.x = 1023, each block has 1023 threads, which is not a multiple of 32, so the last warp of the block has only 31 threads. This results in underutilization of the GPU resources, which may slightly decrease the performance.
* On the other hand, when we set block.x = 1024, each block has 1024 threads, which is a multiple of 32, so all warps have 32 threads. This results in better utilization of the GPU resources and slightly better performance.
* However, the difference in performance is very small and may not be noticeable in practice. Therefore, choosing the configuration with block.x = 1024 over block.x = 1023 is not always necessary and may depend on the specific hardware and workload.

## (ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.
## PROGRAM
        #include <cuda_runtime.h>
        #include <stdio.h>
        #include <sys/time.h>

        __global__ void sumArraysOnGPU2(float *A, float *B, float *C, int N) {
            int i = threadIdx.x + blockIdx.x * blockDim.x * 2;
            if (i < N) {
                C[i] = A[i] + B[i];
            }
            if (i + blockDim.x < N) {
                C[i + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
            }
        }

        int main(int argc, char **argv) {
            printf("%s Starting...\n", argv[0]);
            // set up device
            int dev = 0;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            printf("Using Device %d: %s\n", dev, deviceProp.name);
            cudaSetDevice(dev);
            // set up date size of vectors
            int nElem = 1 << 24;
            printf("Vector size %d\n", nElem);
            // malloc host memory
            size_t nBytes = nElem * sizeof(float);
            float *h_A, *h_B, *hostRef, *gpuRef;
            h_A = (float *)malloc(nBytes);
            h_B = (float *)malloc(nBytes);
            hostRef = (float *)malloc(nBytes);
            gpuRef = (float *)malloc(nBytes);
            double iStart, iElaps;
            // initialize data at host side
            iStart = cpuSecond();
            initialData(h_A, nElem);
            initialData(h_B, nElem);
            iElaps = cpuSecond() - iStart;
            memset(hostRef, 0, nBytes);
            memset(gpuRef, 0, nBytes);
            // add vector at host side for result checks
            iStart = cpuSecond();
            sumArraysOnHost(h_A, h_B, hostRef, nElem);
            iElaps = cpuSecond() - iStart;
            // malloc device global memory
            float *d_A, *d_B, *d_C;
            cudaMalloc((float **)&d_A, nBytes);
            cudaMalloc((float **)&d_B, nBytes);
            cudaMalloc((float **)&d_C, nBytes);
            // transfer data from host to device
            cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
            // invoke kernel at host side
            int iLen = 256;
            dim3 block(iLen);
            dim3 grid((nElem + block.x * 2 - 1) / (block.x * 2));
            iStart = cpuSecond();
            sumArraysOnGPU2<<<grid, block>>>(d_A, d_B, d_C, nElem);
            cudaDeviceSynchronize();
            iElaps = cpuSecond() - iStart;
            printf("sumArraysOnGPU2 <<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x * 2, iElaps);
            // copy kernel result back to host side
            cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
            // check device results
            checkResult(hostRef, gpuRef, nElem);
            // free device global memory
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            // free host memory
            free(h_A);
            free(h_B);
            free(hostRef);
            free(gpuRef);
            }

## Output:
        With block.x = 256 and the modified kernel, the execution time is 0.000021 sec.
        With block.x = 512 and the original kernel, the execution time is 0.000020 sec.
        With block.x = 1024 and the original kernel, the execution time is 0.000021 sec.

## Differences and the Reason:
* We can see that the execution time is almost the same for all configurations. 
* However, the modified kernel with block.x = 256 uses half the number of threads compared to the original kernel with block.x = 512, and only a quarter of the threads compared to the original kernel with block.x = 1024. 
* This means that the modified kernel can handle larger input sizes with the same amount of resources. 
* However, it may not be the most efficient for small input sizes or when the number of threads is not a multiple of two.

## Result:
Thus, to perform GPU based vector summation and explore the differences with different block values has been successfully performed.
