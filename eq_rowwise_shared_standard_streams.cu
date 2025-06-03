// eq_rowwise_shared_streamed.cu
// Shared memory histogram + CUDA Streams strategy

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

// Kernels...

__global__ void rgb2ycbcr_shared(unsigned char* d_image, int width, int height) {
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
    }
}

__global__ void blur_Y(unsigned char* d_image, unsigned char* d_out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int idx = ((row + dy) * width + (col + dx)) * 3;
                sum += d_image[idx];
            }
        int idx = (row * width + col) * 3;
        d_out[idx + 0] = (unsigned char)(sum / 9.0f);
        d_out[idx + 1] = d_image[idx + 1];
        d_out[idx + 2] = d_image[idx + 2];
    }
}

__global__ void histogram_shared(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    __shared__ unsigned int local_hist[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y) local_hist[i] = 0;
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        atomicAdd(&local_hist[d_image[idx]], 1);
    }
    __syncthreads();
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&d_hist[i], local_hist[i]);
}

__global__ void equalize_and_reconstruct(unsigned char* d_image, int* d_cdf, int width, int height) {
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

// MAIN
int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_rowwise_shared_streamed.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) return -1;

    int image_size = width * height * pixelWidth;
    unsigned char *d_image, *d_blur;
    unsigned int* d_hist;
    int* d_cdf;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_blur, image_size);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    // Streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Timers
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    cudaEvent_t start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventCreate(&start2); cudaEventCreate(&stop2);
    cudaEventCreate(&start3); cudaEventCreate(&stop3);
    cudaEventCreate(&start4); cudaEventCreate(&stop4);

    // Step 1: RGB→YCbCr
    cudaEventRecord(start1);
    rgb2ycbcr_shared<<<grid, block, 0, s1>>>(d_image, width, height);
    cudaEventRecord(stop1);

    // Step 2: Blur
    cudaEventRecord(start2);
    blur_Y<<<grid, block, 0, s2>>>(d_image, d_blur, width, height);
    cudaEventRecord(stop2);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaMemcpy(d_image, d_blur, image_size, cudaMemcpyDeviceToDevice);

    // Step 3: Histogram
    cudaEventRecord(start3);
    histogram_shared<<<grid, block>>>(d_image, d_hist, width, height);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    // Step 4: CDF
    unsigned int h_hist[256] = {0};
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int h_cdf[256] = {0}, sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)(((float)(sum - h_hist[0]) / (width * height - 1)) * 255);
    }
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Step 5: Equalize
    cudaEventRecord(start4);
    equalize_and_reconstruct<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);

    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    gettimeofday(&end_total, NULL);
    long sec = end_total.tv_sec - start_total.tv_sec;
    long usec = end_total.tv_usec - start_total.tv_usec;
    double total_time = sec * 1000.0 + usec / 1000.0;

    float t1, t2, t3, t4;
    cudaEventElapsedTime(&t1, start1, stop1);
    cudaEventElapsedTime(&t2, start2, stop2);
    cudaEventElapsedTime(&t3, start3, stop3);
    cudaEventElapsedTime(&t4, start4, stop4);

    printf("\n=== Shared Memory + Streams Report ===\n");
    printf("🔵 RGB → YCbCr (s1): %.3f ms\n", t1);
    printf("🟡 Blur Y (s2)      : %.3f ms\n", t2);
    printf("🟣 Histogram shared : %.3f ms\n", t3);
    printf("🟢 Equalize+RGB     : %.3f ms\n", t4);
    printf("🔷 Total kernel time: %.3f ms\n", t1 + t2 + t3 + t4);
    printf("🕒 Total runtime    : %.3f ms\n", total_time);
    printf("======================================\n");

    // Save and cleanup
    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);
    cudaFree(d_image);
    cudaFree(d_blur);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    return 0;
}
