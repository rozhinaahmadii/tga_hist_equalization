#include <iostream>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#define WIDTH 1920
#define HEIGHT 1080
#define HISTOGRAM_BINS 256

// Kernel amb memòria shared
__global__ void histogram_shared(unsigned char* d_image, int width, int height, int* d_hist) {
    __shared__ unsigned int hist_shared[HISTOGRAM_BINS];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    for (int i = tid; i < HISTOGRAM_BINS; i += threadsPerBlock)
        hist_shared[i] = 0;
    __syncthreads();

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        atomicAdd(&hist_shared[d_image[idx]], 1);
    }

    __syncthreads();

    for (int i = tid; i < HISTOGRAM_BINS; i += threadsPerBlock)
        atomicAdd(&d_hist[i], hist_shared[i]);
}

// Funció per processar en una GPU
void process_on_gpu(int gpu_id, unsigned char* h_image, int width, int height, int* h_hist, float& gpu_time_ms) {
    cudaSetDevice(gpu_id);

    size_t img_size = width * height * sizeof(unsigned char);
    size_t hist_size = HISTOGRAM_BINS * sizeof(int);

    unsigned char* d_image;
    int* d_hist;
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_hist, hist_size);
    cudaMemset(d_hist, 0, hist_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Mesura de temps
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    histogram_shared<<<grid, block>>>(d_image, width, height, d_hist);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    int half_height = height / 2;

    size_t img_size = width * height * sizeof(unsigned char);
    size_t img_half_size = width * half_height * sizeof(unsigned char);
    size_t hist_size = HISTOGRAM_BINS * sizeof(int);

    // Allocat pinned host memory
    unsigned char* h_image;
    unsigned char* h_top;
    unsigned char* h_bottom;
    int* h_hist0;
    int* h_hist1;

    cudaHostAlloc(&h_image, img_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_top, img_half_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_bottom, img_half_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_hist0, hist_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_hist1, hist_size, cudaHostAllocDefault);

    // Inicialització
    for (int i = 0; i < width * height; ++i)
        h_image[i] = static_cast<unsigned char>(i % HISTOGRAM_BINS);

    memset(h_hist0, 0, hist_size);
    memset(h_hist1, 0, hist_size);

    memcpy(h_top, h_image, img_half_size);
    memcpy(h_bottom, h_image + width * half_height, img_half_size);

    float time0 = 0.0f, time1 = 0.0f;

    // Inici del cronòmetre global
    auto t_start = std::chrono::high_resolution_clock::now();

    std::thread t0([&]() {
        process_on_gpu(0, h_top, width, half_height, h_hist0, time0);
    });

    std::thread t1([&]() {
        process_on_gpu(1, h_bottom, width, half_height, h_hist1, time1);
    });

    t0.join();
    t1.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = t_end - t_start;

    // Fusió d’histogrames
    int h_hist[HISTOGRAM_BINS];
    for (int i = 0; i < HISTOGRAM_BINS; ++i)
        h_hist[i] = h_hist0[i] + h_hist1[i];

    // Impressió de temps
    std::cout << "Temps GPU 0: " << time0 << " ms\n";
    std::cout << "Temps GPU 1: " << time1 << " ms\n";
    std::cout << "Temps total (CPU): " << cpu_duration.count() * 1000 << " ms\n";

    // Mostra parcial de l’histograma
    for (int i = 0; i < 16; ++i)
        std::cout << "Bin[" << i << "] = " << h_hist[i] << '\n';

    // Alliberament de memòria
    cudaFreeHost(h_image);
    cudaFreeHost(h_top);
    cudaFreeHost(h_bottom);
    cudaFreeHost(h_hist0);
    cudaFreeHost(h_hist1);

    return 0;
}
