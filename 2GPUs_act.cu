// CUDA 2-GPU Histogram Equalization (Rowwise, Shared Memory, Pinned)

#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define IMG_WIDTH 1920 // Placeholder
#define IMG_HEIGHT 1080 // Placeholder
#define CHANNELS 3
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT * CHANNELS)

__global__ void rgb2ycbcr_rowwise(unsigned char* input, float* y_channel, int width, int height, int offsetY) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y + offsetY;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    y_channel[(y - offsetY) * width + x] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
}

__global__ void histogram_shared(float* y_channel, int* histo, int width, int height) {
    __shared__ int temp[256];
    int tid = threadIdx.x;
    if (tid < 256) temp[tid] = 0;
    __syncthreads();

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    int val = (int)y_channel[y * width + x];
    atomicAdd(&(temp[val]), 1);
    __syncthreads();

    if (tid < 256) atomicAdd(&(histo[tid]), temp[tid]);
}

__global__ void equalize_and_reconstruct(unsigned char* input, unsigned char* output, int* cdf, int width, int height, int offsetY) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y + offsetY;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    for (int i = 0; i < 3; i++) {
        int val = input[idx + i];
        output[idx + i] = (unsigned char)(cdf[val]);
    }
}

void check(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void run_gpu(int device_id, unsigned char* h_input, unsigned char* h_output, int width, int height, int* global_histogram) {
    cudaSetDevice(device_id);
    int half_height = height / 2;
    size_t img_half_size = width * half_height * 3;

    unsigned char *d_input, *d_output;
    float* d_y;
    int* d_hist;

    check(cudaMallocHost((void**)&d_input, img_half_size), "Pinned malloc d_input");
    check(cudaMallocHost((void**)&d_output, img_half_size), "Pinned malloc d_output");

    check(cudaMalloc((void**)&d_y, width * half_height * sizeof(float)), "Malloc d_y");
    check(cudaMalloc((void**)&d_hist, 256 * sizeof(int)), "Malloc d_hist");
    check(cudaMemset(d_hist, 0, 256 * sizeof(int)), "Memset d_hist");

    check(cudaMemcpy(d_input, h_input + device_id * img_half_size, img_half_size, cudaMemcpyHostToDevice), "H2D input");

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x, (half_height + block.y - 1)/block.y);

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rgb2ycbcr_rowwise<<<grid, block>>>(d_input, d_y, width, height, device_id * half_height);
    cudaDeviceSynchronize();

    histogram_shared<<<grid, block>>>(d_y, d_hist, width, half_height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("[GPU %d] Total kernel time: %f ms\n", device_id, elapsed);

    int h_hist[256];
    check(cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost), "D2H hist");
    for (int i = 0; i < 256; i++) global_histogram[i] += h_hist[i];

    cudaFree(d_y);
    cudaFree(d_hist);
    cudaFreeHost(d_input);
    cudaFreeHost(d_output);
}

void compute_cdf(int* hist, int* cdf, int size) {
    cdf[0] = hist[0];
    for (int i = 1; i < size; i++) {
        cdf[i] = cdf[i-1] + hist[i];
    }
    for (int i = 0; i < size; i++) {
        cdf[i] = (cdf[i] * 255) / cdf[255];
    }
}

int main() {
    unsigned char* h_input = (unsigned char*)malloc(IMG_SIZE); // Load actual image data here
    unsigned char* h_output = (unsigned char*)malloc(IMG_SIZE);
    int global_histogram[256] = {0};

    struct timeval start, end;
    gettimeofday(&start, NULL);

    run_gpu(0, h_input, h_output, IMG_WIDTH, IMG_HEIGHT, global_histogram);
    run_gpu(1, h_input, h_output, IMG_WIDTH, IMG_HEIGHT, global_histogram);

    int cdf[256];
    compute_cdf(global_histogram, cdf, 256);

    // Optional: apply equalization in another pass if needed

    gettimeofday(&end, NULL);
    float elapsed = (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f;
    printf("Total runtime: %f ms\n", elapsed);

    free(h_input);
    free(h_output);
    return 0;
}
