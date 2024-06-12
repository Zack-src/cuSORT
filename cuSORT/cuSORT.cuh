#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern __global__ void mergeSortKernel(int* data, int* result, int n);
extern __global__ void quickSortKernel(int* data, int left, int right);
