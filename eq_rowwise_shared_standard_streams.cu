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

__global__ void rgb2ycbcr_rowwise(unsigned char* d_image, unsigned int* d_hist, int width, int height, int row_offset) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_offset;

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

__global__ void blur_Y_channel(unsigned char* d_image, unsigned char* d_blurred, int width, int height, int row_offset) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_offset;

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

__global__ void histogram_shared(unsigned char* d_image, unsigned int* d_hist, int width, int height, int row_offset) {
    __shared__ unsigned int local_hist[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        local_hist[i] = 0;
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_offset;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        unsigned char Y = d_image[idx + 0];
        atomicAdd(&local_hist[Y], 1);
    }

    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&d_hist[i], local_hist[i]);
}

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

int eq_GPU_streams(unsigned char* h_image) {
    int image_size = width * height * pixelWidth;
    unsigned char *d_image, *d_blurred;
    unsigned int *d_hist1, *d_hist2;
    unsigned int h_hist1[256] = {0}, h_hist2[256] = {0};

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_blurred, image_size);
    cudaMalloc((void**)&d_hist1, 256 * sizeof(unsigned int));
    cudaMalloc((void**)&d_hist2, 256 * sizeof(unsigned int));
    cudaMemset(d_hist1, 0, 256 * sizeof(unsigned int));
    cudaMemset(d_hist2, 0, 256 * sizeof(unsigned int));

    // Transfer to device using streams
    cudaMemcpyAsync(d_image, h_image, image_size, cudaMemcpyHostToDevice, stream1);

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height / 2 + 31) / 32); // Half image per stream

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Top half
    rgb2ycbcr_rowwise<<<grid, block, 0, stream1>>>(d_image, d_hist1, width, height, 0);
    blur_Y_channel<<<grid, block, 0, stream1>>>(d_image, d_blurred, width, height, 0);
    histogram_shared<<<grid, block, 0, stream1>>>(d_blurred, d_hist1, width, height, 0);

    // Bottom half
    rgb2ycbcr_rowwise<<<grid, block, 0, stream2>>>(d_image, d_hist2, width, height, height / 2);
    blur_Y_channel<<<grid, block, 0, stream2>>>(d_image, d_blurred, width, height, height / 2);
    histogram_shared<<<grid, block, 0, stream2>>>(d_blurred, d_hist2, width, height, height / 2);

    cudaMemcpyAsync(d_image, d_blurred, image_size, cudaMemcpyDeviceToDevice, stream1);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaMemcpy(h_hist1, d_hist1, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist2, d_hist2, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int h_hist[256];
    for (int i = 0; i < 256; i++)
        h_hist[i] = h_hist1[i] + h_hist2[i];

    // CDF
    int h_cdf[256], sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)((((float)sum - h_hist[0]) / ((float)(width * height - 1))) * 255);
    }

    int* d_cdf;
    cudaMalloc((void**)&d_cdf, 256 * sizeof(int));
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 fullGrid((width + 31) / 32, (height + 31) / 32);
    equalize_and_reconstruct_rowwise<<<fullGrid, block>>>(d_image, d_cdf, width, height);

    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("\n=== Streamed GPU Performance Report (Standard Memory) ===\n");
    printf("🔄 Total kernel + transfer time (streams): %.3f ms\n", elapsed);
    printf("=========================================================\n\n");

    cudaFree(d_image);
    cudaFree(d_blurred);
    cudaFree(d_hist1);
    cudaFree(d_hist2);
    cudaFree(d_cdf);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_standard_streams.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "❌ Couldn't load image.\n");
        return -1;
    }

    printf("📷 Loaded image: %s (W: %d, H: %d, C: %d)\n", input, width, height, pixelWidth);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    eq_GPU_streams(image);
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros  = end.tv_usec - start.tv_usec;
    double elapsed_ms = seconds * 1000.0 + micros / 1000.0;

    printf("✅ Full process time: %.3f ms\n", elapsed_ms);
    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    return 0;
}
