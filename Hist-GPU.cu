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

__global__ void rgb2ycbcr_rowwise(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    int row = blockIdx.x;
    int col = threadIdx.x;

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

        atomicAdd(&(d_hist[Y]), 1);  // now histogram is on GPU
    }
}

__global__ void equalize_and_reconstruct_rowwise(unsigned char* d_image, int* d_cdf, int width, int height) {
    int row = blockIdx.x;
    int col = threadIdx.x;

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
    int image_size = width * height * pixelWidth;
    unsigned char* d_image;
    unsigned int* d_hist;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    dim3 grid(height);
    dim3 block(width);

    // Step 1: RGB → YCbCr and build histogram (row-wise)
    rgb2ycbcr_rowwise<<<grid, block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    // Copy back the modified YCbCr image to CPU
    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);
    stbi_write_png("output_original.png", width, height, pixelWidth, image, 0);

    // Copy histogram back to CPU
    unsigned int h_hist[256] = {0};
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Compute CDF on CPU
    int h_cdf[256] = {0};
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist[i];
        h_cdf[i] = (int)((((float)sum - h_hist[0]) / ((float)(width * height - 1))) * 255);
    }

    // Copy CDF to GPU
    int* d_cdf;
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Step 2: Apply equalization and convert to RGB (row-wise)
    equalize_and_reconstruct_rowwise<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_gpu.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\\n");
        return -1;
    }

    printf("Loaded image: %s (Width: %d, Height: %d, Channels: %d)\\n", input, width, height, pixelWidth);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    eq_GPU(image);

    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros  = end.tv_usec - start.tv_usec;
    double elapsed_ms = seconds * 1000.0 + micros / 1000.0;

    printf("GPU full row-wise histogram equalization done in %.3f ms\\n", elapsed_ms);

    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    return 0;
}
