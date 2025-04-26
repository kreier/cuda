#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel function
__global__ void helloWorldKernel() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch kernel with a single thread
    helloWorldKernel<<<1, 1>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
