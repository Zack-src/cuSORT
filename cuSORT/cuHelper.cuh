
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)


template <typename T>
T* cudaMallocHelper(size_t size) {
    T* var;
    cudaMalloc(&var, size * sizeof(T));
    return var;
}

template <typename T>
void cudaMemcpyToHostHelper(T* hostData, T* deviceData, size_t size) {
    cudaMemcpy(hostData, deviceData, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
void cudaMemcpyHelper(T* dest, const T* src, size_t size, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dest, src, size * sizeof(T), kind));
}
