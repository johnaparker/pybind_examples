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
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
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

    double *partial_c;
    double *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the cpu side
    partial_c = (double*)malloc( blocksPerGrid*sizeof(double) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a,
                              N*sizeof(double) );
    cudaMalloc( (void**)&dev_b,
                              N*sizeof(double) );
    cudaMalloc( (void**)&dev_partial_c,
                              blocksPerGrid*sizeof(double) );

    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );


    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N*sizeof(double),
                              cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N*sizeof(double),
                              cudaMemcpyHostToDevice ); 

    gpu_partial_dot_product<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,
                                            dev_partial_c, N);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,
                                        start, stop );

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( partial_c, dev_partial_c,
                              blocksPerGrid*sizeof(double),
                              cudaMemcpyDeviceToHost );

    // finish up on the CPU side
    double c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // free memory on the gpu side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_partial_c );

    // free memory on the cpu side
    free( partial_c );

    return c;
}
