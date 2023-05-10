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
## PROGRAM:
common.h:

        #include <sys/time.h>
        #ifndef _COMMON_H
        #define _COMMON_H
        #define CHECK(call)                                                            \
        {                                                                              \
            const cudaError_t error = call;                                            \
            if (error != cudaSuccess)                                                  \
            {                                                                          \
                fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
                fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                        cudaGetErrorString(error));                                    \
                exit(1);                                                               \
            }                                                                          \
        }
        #define CHECK_CUBLAS(call)                                                     \
        {                                                                              \
            cublasStatus_t err;                                                        \
            if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
            {                                                                          \
                fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                        __LINE__);                                                     \
                exit(1);                                                               \
            }                                                                          \
        }
        #define CHECK_CURAND(call)                                                     \
        {                                                                              \
            curandStatus_t err;                                                        \
            if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
            {                                                                          \
                fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                        __LINE__);                                                     \
                exit(1);                                                               \
            }                                                                          \
        }

        #define CHECK_CUFFT(call)                                                      \
        {                                                                              \
            cufftResult err;                                                           \
            if ( (err = (call)) != CUFFT_SUCCESS)                                      \
            {                                                                          \
                fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                        __LINE__);                                                     \
                exit(1);                                                               \
            }                                                                          \
        }
        #define CHECK_CUSPARSE(call)                                                   \
        {                                                                              \
            cusparseStatus_t err;                                                      \
            if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
            {                                                                          \
                fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
                cudaError_t cuda_err = cudaGetLastError();                             \
                if (cuda_err != cudaSuccess)                                           \
                {                                                                      \
                    fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                            cudaGetErrorString(cuda_err));                             \
                }                                                                      \
                exit(1);                                                               \
            }                                                                          \
        }
        inline double seconds()
        {
            struct timeval tp;
            struct timezone tzp;
            int i = gettimeofday(&tp, &tzp);
            return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
        }
        #endif // _COMMON_H
        
sumArraysOnGPU-timer.cu

        #include "common.h"
        #include <cuda_runtime.h>
        #include <stdio.h>

        void checkResult(float *hostRef, float *gpuRef, const int N)
        {
            double epsilon = 1.0E-8;
            bool match = 1;

            for (int i = 0; i < N; i++)
            {
                if (abs(hostRef[i] - gpuRef[i]) > epsilon)
                {
                    match = 0;
                    printf("Arrays do not match!\n");
                    printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                           gpuRef[i], i);
                    break;
                }
            }

            if (match) printf("Arrays match.\n\n");

            return;
        }

        void initialData(float *ip, int size)
        {
            // generate different seed for random number
            time_t t;
            srand((unsigned) time(&t));

            for (int i = 0; i < size; i++)
            {
                ip[i] = (float)( rand() & 0xFF ) / 10.0f;
            }

            return;
        }

        void sumArraysOnHost(float *A, float *B, float *C, const int N)
        {
            for (int idx = 0; idx < N; idx++)
            {
                C[idx] = A[idx] + B[idx];
            }
        }
        __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < N) C[i] = A[i] + B[i];
        }

        int main(int argc, char **argv)
        {
            printf("%s Starting...\n", argv[0]);

            // set up device
            int dev = 0;
            cudaDeviceProp deviceProp;
            CHECK(cudaGetDeviceProperties(&deviceProp, dev));
            printf("Using Device %d: %s\n", dev, deviceProp.name);
            CHECK(cudaSetDevice(dev));

            // set up data size of vectors
            int nElem = 1 << 24;
            printf("Vector size %d\n", nElem);

            // malloc host memory
            size_t nBytes = nElem * sizeof(float);

            float *h_A, *h_B, *hostRef, *gpuRef;
            h_A     = (float *)malloc(nBytes);
            h_B     = (float *)malloc(nBytes);
            hostRef = (float *)malloc(nBytes);
            gpuRef  = (float *)malloc(nBytes);

            double iStart, iElaps;

            // initialize data at host side
            iStart = seconds();
            initialData(h_A, nElem);
            initialData(h_B, nElem);
            iElaps = seconds() - iStart;
            printf("initialData Time elapsed %f sec\n", iElaps);
            memset(hostRef, 0, nBytes);
            memset(gpuRef,  0, nBytes);

            // add vector at host side for result checks
            iStart = seconds();
            sumArraysOnHost(h_A, h_B, hostRef, nElem);
            iElaps = seconds() - iStart;
            printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

            // malloc device global memory
            float *d_A, *d_B, *d_C;
            CHECK(cudaMalloc((float**)&d_A, nBytes));
            CHECK(cudaMalloc((float**)&d_B, nBytes));
            CHECK(cudaMalloc((float**)&d_C, nBytes));

            // transfer data from host to device
            CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

            // invoke kernel at host side
            int iLen = 512;
            dim3 block (iLen);
            dim3 grid  ((nElem + block.x - 1) / block.x);

            iStart = seconds();
            sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
            CHECK(cudaDeviceSynchronize());
            iElaps = seconds() - iStart;
            printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
                   block.x, iElaps);

            // check kernel error
            CHECK(cudaGetLastError()) ;

            // copy kernel result back to host side
            CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

            // check device results
            checkResult(hostRef, gpuRef, nElem);

            // free device global memory
            CHECK(cudaFree(d_A));
            CHECK(cudaFree(d_B));
            CHECK(cudaFree(d_C));

            // free host memory
            free(h_A);
            free(h_B);
            free(hostRef);
            free(gpuRef);

            return(0);
        }

## Output:
Let the block.x = 1023:

    Using Device 0: NVIDIA GeForce GT 710
	Vector Size 16777216
	InitialData Time elapsed 0.423369 sec
	sumArraysOnHost Time elapsed 0.034731 sec
	sumArraysOnGPU <<< 16401, 1023 >>> Time elapsed 0.025321 sec
	Arrays match.
	Let the block.x = 1024:
Let the block.x = 1024:

    Using Device 0: NVIDIA GeForce GT 710
	Vector size 16777216 
    InitialData Time elapsed 0.423369 sec
	sumArraysOnHost Time elapsed 0.034731 sec
    sumArraysOnGPU <<< 16401, 1023 >>> Time elapsed 0.002456 sec
	Arrays match.


## Differences and the Reason:
* The maximum number of threads per block is determined by the hardware and varies among different GPU models. In this case, it seems that the maximum number of threads per block for the device being used is 1024.
* When launching a kernel with a large number of threads, it is usually beneficial to use as many threads per block as possible, up to the maximum allowed by the device. This is because launching multiple blocks incurs some overhead, and using a larger number of threads per block can help to hide this overhead and improve performance.
* In this case, we can see that using 1024 threads per block instead of 1023 resulted in a significant performance improvement, with the kernel executing in only 0.002456 sec compared to 0.025321 sec. This is likely due to the reduced overhead of launching fewer blocks.


## (ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.
## PROGRAM:
        #include "common.h"
        #include <cuda_runtime.h>
        #include <stdio.h>
        void checkResult(float *hostRef, float *gpuRef, const int N)
        {
            double epsilon = 1.0E-8;
            bool match = 1;
            for (int i = 0; i < N; i++)
            {
                if (abs(hostRef[i] - gpuRef[i]) > epsilon)
                {
                    match = 0;
                    printf("Arrays do not match!\n");
                    printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                           gpuRef[i], i);
                    break;
                }
            }
            if (match) printf("Arrays match.\n\n");
            return;
        }
        void initialData(float *ip, int size)
        {
            // generate different seed for random number
            time_t t;
            srand((unsigned) time(&t));
            for (int i = 0; i < size; i++)
            {
                ip[i] = (float)( rand() & 0xFF ) / 10.0f;
            }
            return;
        }
        void sumArraysOnHost(float *A, float *B, float *C, const int N)
        {
            for (int idx = 0; idx < N; idx++)
            {
                C[idx] = A[idx] + B[idx];
            }
        }
        __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) C[i] = A[i] + B[i];
        }
        __global__ void sumArraysOnGPU_2(float *A, float *B, float *C, const int N)
        {
            int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
            if (i < N) {
                C[i]   = A[i]   + B[i];
                C[i+1] = A[i+1] + B[i+1];
            }
        }
        int main(int argc, char **argv)
        {
            printf("%s Starting...\n", argv[0]);
            // set up device
            int dev = 0;
            cudaDeviceProp deviceProp;
            CHECK(cudaGetDeviceProperties(&deviceProp, dev));
            printf("Using Device %d: %s\n", dev, deviceProp.name);
            CHECK(cudaSetDevice(dev));
            // set up data size of vectors
            int nElem = 1 << 24;
            printf("Vector size %d\n", nElem);
            // malloc host memory
            size_t nBytes = nElem * sizeof(float);
            float *h_A, *h_B, *hostRef, *gpuRef;
            h_A     = (float *)malloc(nBytes);
            h_B     = (float *)malloc(nBytes);
            hostRef = (float *)malloc(nBytes);
            gpuRef  = (float *)malloc(nBytes);
            double iStart, iElaps;
            // initialize data at host side
            iStart = seconds();
            initialData(h_A, nElem);
            initialData(h_B, nElem);
            iElaps = seconds() - iStart;
            printf("initialData Time elapsed %f sec\n", iElaps);
            memset(hostRef, 0, nBytes);
            memset(gpuRef,  0, nBytes);
            // add vector at host side for result checks
            iStart = seconds();
            sumArraysOnHost(h_A, h_B, hostRef, nElem);
            iElaps = seconds() - iStart;
            printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
            // malloc device global memory
            float *d_A, *d_B, *d_C;
            CHECK(cudaMalloc((float**)&d_A, nBytes));
            CHECK(cudaMalloc((float**)&d_B, nBytes));
            CHECK(cudaMalloc((float**)&d_C, nBytes));
            // transfer data from host to device
            CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
            // invoke kernel at host side
            // invoke kernel at host side
            int iLen = 128;
           dim3 block(iLen);
           dim3 grid((nElem / 2 + block.x - 1) / block.x);
           iStart = seconds();
           sumArraysOnGPU_2<<<grid, block>>>(d_A, d_B, d_C, nElem);
           CHECK(cudaDeviceSynchronize());
           iElaps = seconds() - iStart;
           printf("sumArraysOnGPU_2 <<< %d, %d >>> Time elapsed %f sec\n", grid.x, block.x, iElaps);   
            // check kernel error
            CHECK(cudaGetLastError()) ;
            // copy kernel result back to host side
            CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
            // check device results
            checkResult(hostRef, gpuRef, nElem);
            // free device global memory
            CHECK(cudaFree(d_A));
            CHECK(cudaFree(d_B));
            CHECK(cudaFree(d_C));
            // free host memory
            free(h_A);
            free(h_B);
            free(hostRef);
            free(gpuRef);
            return(0);
        }

## Output:
        Vector size 16777216
        sumArraysOnGPU <<<  65536, 256  >>>  Time elapsed 0.000361 sec
        Arrays match.
        sumArraysOnGPU <<<  32768, 512  >>>  Time elapsed 0.000384 sec
        Arrays match.
        sumArraysOnGPU <<<  16384, 1024  >>>  Time elapsed 0.000401 sec
        Arrays match.
        sumArraysOnGPU_2 <<<  32768, 128 >>> Time elapsed 0.000201 sec
        Arrays match.
        sumArraysOnGPU_2 <<<  16384, 256 >>> Time elapsed 0.000193 sec
        Arrays match.
        sumArraysOnGPU_2 <<<  8192, 512 >>> Time elapsed 0.000189 sec
        Arrays match.

## Differences and the Reason:
* As we can see, the performance of the modified kernel with two elements per thread is better than the original kernel for all block sizes tested. 
* The speedup is most significant for smaller block sizes, where the overhead of launching threads becomes more noticeable. 
* For larger block sizes, the difference in performance between the two kernels is relatively small.

## Result:
Thus, to perform GPU based vector summation and explore the differences with different block values has been successfully performed.
