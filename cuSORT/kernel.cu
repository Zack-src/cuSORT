
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

    int n = 1000000000;
    int* array = new int[n];
    
    for(int i = 0; i < n; ++i) {
        array[i] = rand() % n;
    }

    int* d_array;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    auto start = std::chrono::high_resolution_clock::now();

    //mergeSortKernel <<<numBlocks, blockSize>>> (d_array, d_temp, n);
    quickSortKernel <<<1, 1>>> (d_array, 0, n - 1);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "time : " << duration << " ms" << std::endl;

    //cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < n; ++i) {
    //    std::cout << array[i] << " ";
    //}
    //std::cout << std::endl;

    cudaFree(d_array);
    delete[] array;
}