// eq_rowwise_shared_streamed.cu
// Shared memory histogram + CUDA Streams strategy (based on existing shared standard code)

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

// Kernel: RGB → YCbCr
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

// Blur Y
__global__ void blur_Y(unsigned char* d_image, unsigned char* d_out, int width, int height) {
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
        int idx = (row * width + col) * 3;
        d_out[idx + 0] = (unsigned char)(sum / 9.0f);
        d_out[idx + 1] = d_image[idx + 1];
        d_out[idx + 2] = d_image[idx + 2];
    }
}

// Shared memory histogram
__global__ void histogram_shared(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    __shared__ unsigned int local_hist[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y) local_hist[i] = 0;
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        atomicAdd(&local_hist[d_image[idx + 0]], 1);
    }
    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&d_hist[i], local_hist[i]);
}

// Reconstruct
__global__ void equalize_and_reconstruct(unsigned char* d_image, int* d_cdf, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        int Y = d_image[idx + 0];
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

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_shared_streamed.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) return -1;

    int image_size = width * height * pixelWidth;
    unsigned char *d_image, *d_blur;
    unsigned int* d_hist;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_blur, image_size);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    int* d_cdf;
    cudaMalloc(&d_cdf, 256 * sizeof(int));

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    rgb2ycbcr_shared<<<grid, block, 0, stream1>>>(d_image, width, height);
    blur_Y<<<grid, block, 0, stream2>>>(d_image, d_blur, width, height);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaMemcpy(d_image, d_blur, image_size, cudaMemcpyDeviceToDevice);
    histogram_shared<<<grid, block>>>(d_image, d_hist, width, height);

    unsigned int h_hist[256] = {0};
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int h_cdf[256] = {0}, sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)((((float)sum - h_hist[0]) / ((float)(width * height - 1))) * 255);
    }
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    equalize_and_reconstruct<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    cudaFree(d_image);
    cudaFree(d_blur);
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
