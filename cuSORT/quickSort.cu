#include "cuSORT.cuh"

__device__ int partition(int* data, int left, int right) {
    int pivot = data[right];
    int i = (left - 1);

    for (int j = left; j <= right - 1; j++) {
        if (data[j] < pivot) {
            i++;
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    int temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;
    return (i + 1);
}

__global__ void quickSortKernel(int* data, int left, int right) {
    if (left < right) {
        int pi = partition(data, left, right);

#ifdef USE_DYNAMIC_PARALLELISM
        if (left < pi - 1)
            quickSortKernel << <1, 1 >> > (data, left, pi - 1);

        if (pi + 1 < right)
            quickSortKernel << <1, 1 >> > (data, pi + 1, right);
#else
    # if __CUDA_ARCH__>=200
        printf("Dynamic parallelism not supported.\n");
    #endif  
#endif
    }
}