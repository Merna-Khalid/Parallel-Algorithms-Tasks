#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector>

using namespace std;


__global__ void distanceKernel(int* a, int m, int n, int* d)
{
    int row1 = blockIdx.x * blockDim.x + threadIdx.x;
    int row2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (row1 < m && row2 < m)
    {
        int tmp = 0;
        for (int i = 0; i < n; i++)
        {
            tmp += (a[row1 * n + i] - a[row2 * n + i]) * (a[row1 * n + i] - a[row2 * n + i]);
        }
        d[row1 * m + row2] = tmp;
    }
}

void distance(int* a, int m, int n, int* d)
{
    dim3 threadsPerBlock(m, m);
    dim3 blocksPerGrid(1, 1);
    if (m > 16) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 16;
        blocksPerGrid.x = ceil(double(m) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(m) / double(threadsPerBlock.y));
        printf("%d\n", blocksPerGrid.x);
    }

    distanceKernel << <blocksPerGrid, threadsPerBlock >> > (a, m, n, d);
    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    }
}

int main(int argc, char* argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char**)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    //int m = 1024;
    //int n = 512;
    int m = 128;
    int n = 16;
    int nmbytes = n * m * sizeof(int);
    int mmbytes = m * m * sizeof(int);

    // allocate host memory
    int* a = 0;
    checkCudaErrors(cudaMallocHost((void**)&a, nmbytes));
    memset(a, 0, nmbytes);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * m + j] = 6;
    }

    int* d = 0;
    checkCudaErrors(cudaMallocHost((void**)&d, mmbytes));
    memset(d, 0, mmbytes);


    // allocate device memory
    int* d_a = 0;
    checkCudaErrors(cudaMalloc((void**)&d_a, nmbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nmbytes));

    int* d_d = 0;
    checkCudaErrors(cudaMalloc((void**)&d_d, mmbytes));
    checkCudaErrors(cudaMemset(d_d, 255, mmbytes));

    // set kernel launch configuration
    //dim3 threads = dim3(512, 1);
    //dim3 blocks  = dim3(m * n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    cudaMemcpyAsync(d_a, a, nmbytes, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_d, d, mmbytes, cudaMemcpyHostToDevice, 0);

    distance(d_a, m, n, d_d);
    cudaMemcpyAsync(a, d_a, nmbytes, cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(d, d_d, mmbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long long counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    //bool bFinalResults = correct_output(a, n, value);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
            printf("%d ", d[i * m + j]);
        printf("\n");
    }

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFreeHost(d));
    checkCudaErrors(cudaFree(d_d));

    return 0;
}