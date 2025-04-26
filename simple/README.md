# Simple CUDA programs

## Hello world!

Compile it with `nvcc hello_world.cu -o hello_world`. Then execute it with `./hello_world`

The created binaries vary in size:

- V10.2.300 on Jetson Nano 620,880 Bytes
- 

## Prime

With just C++ it needs 4.868 s to find primes until 100,000,000. Using the 128 CUDA cores of the Maxwell GPU in the Jetson Nano this time is reduced to 0.766 ms - 44x faster!

``` cpp
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define MAX 100000000

void sieve(bool primes[]) {
    for (int i = 2; i < MAX; i++) primes[i] = true;

    for (int i = 2; i <= sqrt(MAX); i++) {
        if (primes[i]) {
            for (int j = i * i; j < MAX; j += i)
                primes[j] = false;
        }
    }
}

int main() {
//    bool primes[MAX] = {false}; // memory segmentation error above 8 million
    bool *primes = (bool*)malloc(MAX * sizeof(bool));
    if (!primes) {
        printf("Memory allocation failed!\n");
        return -1;
    }  
    int prime_count = 0;
    clock_t start, end;
    double cpu_time_used;

    printf("Calculating prime numbers until %d using Sieve of Eratosthenes...\n", MAX);
    
    start = clock();
    sieve(primes);
    end = clock();
    
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Count prime numbers
    for (int i = 2; i < MAX; i++) {
        if (primes[i]) prime_count++;
    }

    printf("Total number of primes found: %d\n", prime_count);
    printf("Execution Time: %f seconds.\n", cpu_time_used);

    free(primes);
    return 0;
}
```

And here is the CUDA version:

``` cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define MAX 10000000
#define BLOCK_SIZE 256

// CUDA kernel for marking non-primes
__global__ void sieveKernel(bool *primes, int sqrt_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i > sqrt_max || !primes[i]) return;

    for (int j = i * i; j < MAX; j += i) {
        primes[j] = false;
    }
}

int main() {
    bool *d_primes, h_primes[MAX];
    int prime_count = 0;
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_primes, MAX * sizeof(bool));

    // Initialize all numbers as prime
    for (int i = 2; i < MAX; i++) h_primes[i] = true;
    
    cudaMemcpy(d_primes, h_primes, MAX * sizeof(bool), cudaMemcpyHostToDevice);

    printf("Calculating prime numbers until %d using CUDA...\n", MAX);
    
    // Start timer
    start = clock();

    // Launch CUDA kernel
    int sqrt_max = sqrt(MAX);
    sieveKernel<<<(sqrt_max + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_primes, sqrt_max);
    cudaDeviceSynchronize();

    // Stop timer
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Copy results back to CPU
    cudaMemcpy(h_primes, d_primes, MAX * sizeof(bool), cudaMemcpyDeviceToHost);

    // Count primes
    for (int i = 2; i < MAX; i++) {
        if (h_primes[i]) prime_count++;
    }

    printf("Total number of primes found: %d\n", prime_count);
    printf("Execution Time: %f seconds.\n", cpu_time_used);

    // Free GPU memory
    cudaFree(d_primes);

    return 0;
}
```
