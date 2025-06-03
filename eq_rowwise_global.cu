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

// Kernel: RGB → YCbCr and initial histogram
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

// Blur Y channel only
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
                sum += d_image[idx + 0];  // Y
            }
        }
        int out_idx = (row * width + col) * 3;
        d_blurred[out_idx + 0] = (unsigned char)(sum / 9.0f);      // blurred Y
        d_blurred[out_idx + 1] = d_image[out_idx + 1];             // Cb
        d_blurred[out_idx + 2] = d_image[out_idx + 2];             // Cr
    }
}

// Global memory histogram kernel
__global__ void histogram_global(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        unsigned char Y = d_image[idx + 0];
        atomicAdd(&d_hist[Y], 1);
    }
}

// Equalize and convert back to RGB
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

// GPU pipeline
int eq_GPU(unsigned char* image) {
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

    // Timers
    cudaEvent_t startYCbCr, stopYCbCr;
    cudaEvent_t startBlur, stopBlur;
    cudaEvent_t startHist, stopHist;
    cudaEvent_t startEqualize, stopEqualize;

    cudaEventCreate(&startYCbCr);    cudaEventCreate(&stopYCbCr);
    cudaEventCreate(&startBlur);     cudaEventCreate(&stopBlur);
    cudaEventCreate(&startHist);     cudaEventCreate(&stopHist);
    cudaEventCreate(&startEqualize); cudaEventCreate(&stopEqualize);

    // Step 1: RGB → YCbCr
    cudaEventRecord(startYCbCr);
    rgb2ycbcr_rowwise<<<grid, block>>>(d_image, d_hist, width, height);
    cudaEventRecord(stopYCbCr);
    cudaEventSynchronize(stopYCbCr);

    // Step 2: Blur
    cudaEventRecord(startBlur);
    blur_Y_channel<<<grid, block>>>(d_image, d_blurred, width, height);
    cudaEventRecord(stopBlur);
    cudaEventSynchronize(stopBlur);
    cudaMemcpy(d_image, d_blurred, image_size, cudaMemcpyDeviceToDevice);

    // Step 3: Histogram (global memory)
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    cudaEventRecord(startHist);
    histogram_global<<<grid, block>>>(d_image, d_hist, width, height);
    cudaEventRecord(stopHist);
    cudaEventSynchronize(stopHist);

    // Step 4: CDF on CPU
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

    // Step 5: Equalize and reconstruct
    cudaEventRecord(startEqualize);
    equalize_and_reconstruct_rowwise<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaEventRecord(stopEqualize);
    cudaEventSynchronize(stopEqualize);

    // Copy back
    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    // Timing results
    float tYCbCr = 0, tBlur = 0, tHist = 0, tEq = 0;
    cudaEventElapsedTime(&tYCbCr, startYCbCr, stopYCbCr);
    cudaEventElapsedTime(&tBlur, startBlur, stopBlur);
    cudaEventElapsedTime(&tHist, startHist, stopHist);
    cudaEventElapsedTime(&tEq, startEqualize, stopEqualize);

    printf("\n=== Kernel Performance Report ===\n");
    printf("🔵 RGB → YCbCr + initial hist: %.3f ms\n", tYCbCr);
    printf("🟡 Blur Y channel           : %.3f ms\n", tBlur);
    printf("🟣 Histogram (global mem)   : %.3f ms\n", tHist);
    printf("🟢 Equalize + Reconstruct   : %.3f ms\n", tEq);
    printf("🔷 Total kernel time        : %.3f ms\n", tYCbCr + tBlur + tHist + tEq);
    printf("=================================\n\n");

    cudaFree(d_image);
    cudaFree(d_blurred);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_equalized_gpu_global.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return -1;
    }

    printf("Loaded image: %s (Width: %d, Height: %d, Channels: %d)\n", input, width, height, pixelWidth);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    eq_GPU(image);

    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros  = end.tv_usec - start.tv_usec;
    double elapsed_ms = seconds * 1000.0 + micros / 1000.0;

    printf("✅ GPU histogram equalization with blur (global mem) done in %.3f ms\n", elapsed_ms);

    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    return 0;
}
