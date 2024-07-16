
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include "cuSORT.cuh"
#include "cuHelper.cuh"

int main()
{
    srand(time(NULL));

    int n = 100;
    float* array = new float[n];
    
    for(int i = 0; i < n; ++i) {
        array[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / n));
    }

    float* d_array;
    cudaMalloc(&d_array, n * sizeof(float));
    cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_temp;
    cudaMalloc(&d_temp, n * sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mergeSortKernel <<<numBlocks, blockSize>>> (d_array, d_temp, n);

    cudaMemcpy(array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        std::cout << array[i] << (array[i] >= array[i-1] ? "\n" : "| superior\n");
    }
    std::cout << std::endl;

    cudaFree(d_array);
    delete[] array;
}