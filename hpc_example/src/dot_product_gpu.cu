#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 256;

__global__ void gpu_partial_dot_product( double *a, double *b, double *c, int N) {
    __shared__ double cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

double gpu_full_dot_product(const double *a, const double *b, int N) {
    const int blocksPerGrid = imin( 256, (N+threadsPerBlock-1) / threadsPerBlock );

    double *partial_sum;
    double *dev_a, *dev_b, *dev_partial_sum;

    // allocate memory on the cpu side
    partial_sum = (double*)malloc( blocksPerGrid*sizeof(double) );

    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_a, N*sizeof(double));
    cudaMalloc((void**)&dev_b, N*sizeof(double));
    cudaMalloc((void**)&dev_partial_sum, blocksPerGrid*sizeof(double));

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( dev_b, b, N*sizeof(double), cudaMemcpyHostToDevice); 

    gpu_partial_dot_product<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b,
                                            dev_partial_sum, N);

    cudaMemcpy(partial_sum, dev_partial_sum,
                    blocksPerGrid*sizeof(double),
                    cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        sum += partial_sum[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_sum);

    // free memory on the cpu side
    free(partial_sum);

    return sum;
}
