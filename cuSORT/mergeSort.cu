
#include "cuSORT.cuh"

__device__ void merge(int* data, int start, int mid, int end, int* result) {
    int i = start, j = mid + 1, k = start;
    while (i <= mid && j <= end) {
        if (data[i] <= data[j]) {
            result[k++] = data[i++];
        }
        else {
            result[k++] = data[j++];
        }
    }
    while (i <= mid) {
        result[k++] = data[i++];
    }
    while (j <= end) {
        result[k++] = data[j++];
    }
    for (i = start; i <= end; i++) {
        data[i] = result[i];
    }
}

__global__ void mergeSortKernel(int* data, int* result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int current_size = 2;
    int start, mid, end;
    while (current_size <= n) {
        if (index % current_size == 0) {
            start = index;
            mid = min(start + current_size / 2 - 1, n - 1);
            end = min(start + current_size - 1, n - 1);
            merge(data, start, mid, end, result);
        }
        __syncthreads(); // Synchronize threads
        current_size *= 2;
    }
}