// eq_rowwise_shared_streamed_final.cu
// Optimized with CUDA Streams for parallelism and correctness
// Strategy Summary:
// 1. Split image into overlapping tiles for safe blur.
// 2. Launch RGB to YCbCr and blur per tile using separate streams.
// 3. Synchronize all streams.
// 4. Compute a single global histogram and CDF.
// 5. Launch equalization and RGB reconstruction per tile in streams.
// 6. Record performance using CUDA Events + std::chrono for total runtime and detailed kernel timings.

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

unsigned char *image;
int width, height, pixelWidth;

// CUDA kernels
__global__ void rgb2ycbcr_kernel(unsigned char* img, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * 3;
        int r = img[idx + 0];
        int g = img[idx + 1];
        int b = img[idx + 2];

        int Y  = (int)(16 + 0.25679890625 * r + 0.50412890625 * g + 0.09790625 * b);
        int Cb = (int)(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
        int Cr = (int)(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

        img[idx + 0] = Y;
        img[idx + 1] = Cb;
        img[idx + 2] = Cr;
    }
}

__global__ void blur_kernel(unsigned char* img, unsigned char* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int x = col + dx;
                int y = row + dy;
                int idx = (y * width + x) * 3;
                sum += img[idx + 0];
            }
        }
        int out_idx = (row * width + col) * 3;
        out[out_idx + 0] = (unsigned char)(sum / 9.0f);
        out[out_idx + 1] = img[out_idx + 1];
        out[out_idx + 2] = img[out_idx + 2];
    }
}

__global__ void histogram_kernel(unsigned char* img, unsigned int* hist, int width, int height) {
    __shared__ unsigned int local_hist[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        local_hist[i] = 0;
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * 3;
        unsigned char Y = img[idx];
        atomicAdd(&local_hist[Y], 1);
    }
    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&hist[i], local_hist[i]);
}

__global__ void equalize_kernel(unsigned char* img, int* cdf, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * 3;
        int Y  = img[idx + 0];
        int Cb = img[idx + 1];
        int Cr = img[idx + 2];

        int new_Y = cdf[Y];
        int R = min(255, max(0, (int)(new_Y + 1.402 * (Cr - 128))));
        int G = min(255, max(0, (int)(new_Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
        int B = min(255, max(0, (int)(new_Y + 1.772 * (Cb - 128))));

        img[idx + 0] = R;
        img[idx + 1] = G;
        img[idx + 2] = B;
    }
}

// Host function
void process_image_with_streams(unsigned char* h_image) {
    size_t imgSize = width * height * pixelWidth;
    unsigned char *d_img, *d_blur;
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_blur, imgSize);
    cudaMemcpy(d_img, h_image, imgSize, cudaMemcpyHostToDevice);

    int streamCount = 2;
    int overlap = 1;
    int tileHeight = height / streamCount;

    cudaStream_t streams[streamCount];
    for (int i = 0; i < streamCount; i++) cudaStreamCreate(&streams[i]);

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (tileHeight + 1 + 31) / 32);

    cudaEvent_t t1, t2, t3, t4;
    cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3); cudaEventCreate(&t4);
    cudaEventRecord(t1);

    for (int i = 0; i < streamCount; i++) {
        int yOffset = i * tileHeight - (i > 0 ? overlap : 0);
        int actualHeight = tileHeight + (i > 0 ? overlap : 0) + (i < streamCount - 1 ? overlap : 0);
        unsigned char* tileStart = d_img + yOffset * width * 3;
        unsigned char* tileOut = d_blur + yOffset * width * 3;

        dim3 gridTile((width + 31) / 32, (actualHeight + 31) / 32);
        rgb2ycbcr_kernel<<<gridTile, block, 0, streams[i]>>>(tileStart, width, actualHeight);
        blur_kernel<<<gridTile, block, 0, streams[i]>>>(tileStart, tileOut, width, actualHeight);
    }

    for (int i = 0; i < streamCount; i++) cudaStreamSynchronize(streams[i]);
    cudaEventRecord(t2);

    unsigned int* d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    histogram_kernel<<<dim3((width + 31) / 32, (height + 31) / 32), block>>>(d_blur, d_hist, width, height);
    cudaEventRecord(t3);

    unsigned int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int h_cdf[256], sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)((((float)sum - h_hist[0]) / ((float)(width * height - 1))) * 255);
    }

    int* d_cdf;
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < streamCount; i++) {
        int yOffset = i * tileHeight;
        int actualHeight = tileHeight;
        dim3 gridTile((width + 31) / 32, (actualHeight + 31) / 32);
        equalize_kernel<<<gridTile, block, 0, streams[i]>>>(d_blur + yOffset * width * 3, d_cdf, width, actualHeight);
    }

    for (int i = 0; i < streamCount; i++) cudaStreamSynchronize(streams[i]);
    cudaEventRecord(t4);

    cudaMemcpy(h_image, d_blur, imgSize, cudaMemcpyDeviceToHost);

    float dt1, dt2, dt3, dt4;
    cudaEventElapsedTime(&dt1, t1, t2);
    cudaEventElapsedTime(&dt2, t2, t3);
    cudaEventElapsedTime(&dt3, t3, t4);
    cudaEventElapsedTime(&dt4, t1, t4);

    printf("\n=== Shared Memory + Streams Report ===\n");
    printf("🔵 RGB → YCbCr + Blur : %.3f ms\n", dt1);
    printf("🟣 Histogram time     : %.3f ms\n", dt2);
    printf("🟢 Equalize time      : %.3f ms\n", dt3);
    printf("🔷 Total kernel time  : %.3f ms\n", dt4);
    for (int i = 0; i < streamCount; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_img); cudaFree(d_blur); cudaFree(d_hist); cudaFree(d_cdf);
}

// MAIN
int main() {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_final_streamed.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "❌ Couldn't load image.\n");
        return -1;
    }

    printf("📷 Loaded image: %s (W: %d, H: %d, C: %d)\n", input, width, height, pixelWidth);

    auto start = chrono::high_resolution_clock::now();
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);
    process_image_with_streams(image);
    auto end = chrono::high_resolution_clock::now();
    gettimeofday(&end_total, NULL);
    long sec = end_total.tv_sec - start_total.tv_sec;
    long usec = end_total.tv_usec - start_total.tv_usec;
    double total_time = sec * 1000.0 + usec / 1000.0;    printf("🕒 Total runtime    : %.3f ms\n", total_time);
    chrono::duration<double, std::milli> elapsed = end - start;
    printf("✅ Full process time   : %.3f ms\n", elapsed.count());
    printf("======================================\n\n");

    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    return 0;
}
