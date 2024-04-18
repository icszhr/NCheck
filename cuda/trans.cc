#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

void testTransfer(float* h_a, float* d_a, int size, const char* testName) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 主机到设备
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << testName << " - Host to Device transfer time: " << milliseconds << " ms" << std::endl;

    // 设备到主机
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(h_a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << testName << " - Device to Host transfer time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

int main() {
    const int size = 1 << 24;
    float *h_a_pinned, *h_a_pageable, *d_a;

    // 分配可锁定主机内存和设备内存
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_a_pinned, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, size * sizeof(float)));
    // 分配可分页主机内存
    h_a_pageable = new float[size];

    // 初始化数据
    for(int i = 0; i < size; i++) {
        h_a_pinned[i] = h_a_pageable[i] = static_cast<float>(i);
    }

    // 测试可锁定内存传输性能
    testTransfer(h_a_pinned, d_a, size, "Pinned Memory");

    // 测试可分页内存传输性能
    testTransfer(h_a_pageable, d_a, size, "Pageable Memory");

    // 清理资源
    CHECK_CUDA_ERROR(cudaFreeHost(h_a_pinned));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    delete[] h_a_pageable;

    return 0;
}
