#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

unsigned char *image;
int width, height, pixelWidth;

// Kernel: RGB → YCbCr + initial histogram
__global__ void rgb2ycbcr_rowwise(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        int r = d_image[idx + 0];
        int g = d_image[idx + 1];
        int b = d_image[idx + 2];

        int Y  = (int)(16 + 0.25679890625 * r + 0.50412890625 * g + 0.09790625 * b);
        int Cb = (int)(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
        int Cr = (int)(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

        d_image[idx + 0] = Y;
        d_image[idx + 1] = Cb;
        d_image[idx + 2] = Cr;

        atomicAdd(&(d_hist[Y]), 1);
    }
}

// Blur only Y channel
__global__ void blur_Y_channel(unsigned char* d_image, unsigned char* d_blurred, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int x = col + dx;
                int y = row + dy;
                int idx = (y * width + x) * 3;
                sum += d_image[idx + 0];
            }
        }
        int out_idx = (row * width + col) * 3;
        d_blurred[out_idx + 0] = (unsigned char)(sum / 9.0f);
        d_blurred[out_idx + 1] = d_image[out_idx + 1];
        d_blurred[out_idx + 2] = d_image[out_idx + 2];
    }
}

// Histogram with shared memory
__global__ void histogram_shared(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    __shared__ unsigned int local_hist[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        local_hist[i] = 0;

    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        unsigned char Y = d_image[idx + 0];
        atomicAdd(&local_hist[Y], 1);
    }

    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&d_hist[i], local_hist[i]);
}

// Equalize + reconstruct to RGB
__global__ void equalize_and_reconstruct_rowwise(unsigned char* d_image, int* d_cdf, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        int Y  = d_image[idx + 0];
        int Cb = d_image[idx + 1];
        int Cr = d_image[idx + 2];

        int new_Y = d_cdf[Y];

        int R = min(255, max(0, (int)(new_Y + 1.402 * (Cr - 128))));
        int G = min(255, max(0, (int)(new_Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
        int B = min(255, max(0, (int)(new_Y + 1.772 * (Cb - 128))));

        d_image[idx + 0] = R;
        d_image[idx + 1] = G;
        d_image[idx + 2] = B;
    }
}

int eq_GPU(unsigned char* image) {
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    int image_size = width * height * pixelWidth;
    unsigned char *d_image, *d_blurred;
    unsigned int* d_hist;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_blurred, image_size);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t startYCbCr, stopYCbCr;
    cudaEvent_t startBlur, stopBlur;
    cudaEvent_t startHist, stopHist;
    cudaEvent_t startEqualize, stopEqualize;

    cudaEventCreate(&startYCbCr);    cudaEventCreate(&stopYCbCr);
    cudaEventCreate(&startBlur);     cudaEventCreate(&stopBlur);
    cudaEventCreate(&startHist);     cudaEventCreate(&stopHist);
    cudaEventCreate(&startEqualize); cudaEventCreate(&stopEqualize);

    // Step 1
    cudaEventRecord(startYCbCr);
    rgb2ycbcr_rowwise<<<grid, block>>>(d_image, d_hist, width, height);
    cudaEventRecord(stopYCbCr);
    cudaEventSynchronize(stopYCbCr);

    // Step 2
    cudaEventRecord(startBlur);
    blur_Y_channel<<<grid, block>>>(d_image, d_blurred, width, height);
    cudaEventRecord(stopBlur);
    cudaEventSynchronize(stopBlur);
    cudaMemcpy(d_image, d_blurred, image_size, cudaMemcpyDeviceToDevice);

    // Step 3
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    cudaEventRecord(startHist);
    histogram_shared<<<grid, block>>>(d_image, d_hist, width, height);
    cudaEventRecord(stopHist);
    cudaEventSynchronize(stopHist);

    // Step 4: CPU CDF
    unsigned int h_hist[256] = {0};
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int h_cdf[256] = {0}, sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)((((float)sum - h_hist[0]) / ((float)(width * height - 1))) * 255);
    }

    int* d_cdf;
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Step 5
    cudaEventRecord(startEqualize);
    equalize_and_reconstruct_rowwise<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaEventRecord(stopEqualize);
    cudaEventSynchronize(stopEqualize);

    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    float tYCbCr = 0, tBlur = 0, tHist = 0, tEq = 0;
    cudaEventElapsedTime(&tYCbCr, startYCbCr, stopYCbCr);
    cudaEventElapsedTime(&tBlur, startBlur, stopBlur);
    cudaEventElapsedTime(&tHist, startHist, stopHist);
    cudaEventElapsedTime(&tEq, startEqualize, stopEqualize);

    printf("\n=== Kernel Performance Report ===\n");
    printf("🔵 RGB → YCbCr + initial hist: %.3f ms\n", tYCbCr);
    printf("🟡 Blur Y channel           : %.3f ms\n", tBlur);
    printf("🟣 Histogram (shared mem)   : %.3f ms\n", tHist);
    printf("🟢 Equalize + Reconstruct   : %.3f ms\n", tEq);
    printf("🔷 Total kernel time        : %.3f ms\n", tYCbCr + tBlur + tHist + tEq);
    printf("=================================\n");

    gettimeofday(&end_total, NULL);
    long sec = end_total.tv_sec - start_total.tv_sec;
    long usec = end_total.tv_usec - start_total.tv_usec;
    double total_time_ms = sec * 1000.0 + usec / 1000.0;
    printf("🕒 Total GPU execution time (gettimeofday): %.3f ms\n\n", total_time_ms);

    cudaFree(d_image);
    cudaFree(d_blurred);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_equalized_gpu_pinned.png";

    int n_channels;
    unsigned char* raw = stbi_load(input, &width, &height, &n_channels, 0);
    if (!raw) {
        fprintf(stderr, "❌ Couldn't load image.\n");
        return -1;
    }

    pixelWidth = n_channels;
    int size = width * height * pixelWidth;

    // ✅ Allocate pinned memory and copy image
    cudaHostAlloc((void**)&image, size, cudaHostAllocDefault);
    memcpy(image, raw, size);
    stbi_image_free(raw); // Free raw image right after copy

    printf("📷 Loaded image: %s (Width: %d, Height: %d, Channels: %d)\n", input, width, height, pixelWidth);

    // ✅ Measure total runtime *after* data is in pinned memory
    struct timeval start_main, end_main;
    gettimeofday(&start_main, NULL);

    eq_GPU(image);               // GPU execution
    cudaDeviceSynchronize();     // ✅ Wait for everything

    gettimeofday(&end_main, NULL);
    long sec = end_main.tv_sec - start_main.tv_sec;
    long usec = end_main.tv_usec - start_main.tv_usec;
    double elapsed_ms = sec * 1000.0 + usec / 1000.0;
    printf("✅ Total main() runtime (incl. GPU): %.3f ms\n", elapsed_ms);

    // Save and cleanup
    stbi_write_png(output, width, height, pixelWidth, image, 0);
    cudaFreeHost(image);

    return 0;
}

