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

__global__ void rgb2ycbcr_kernel(unsigned char* d_image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < width * height; i += stride) {
        int r = d_image[3*i + 0];
        int g = d_image[3*i + 1];
        int b = d_image[3*i + 2];

        int Y = (int)(16 + 0.25679890625*r + 0.50412890625*g + 0.09790625*b);
        int Cb = (int)(128 - 0.168736*r - 0.331264*g + 0.5*b);
        int Cr = (int)(128 + 0.5*r - 0.418688*g - 0.081312*b);

        d_image[3*i + 0] = Y;
        d_image[3*i + 1] = Cb;
        d_image[3*i + 2] = Cr;
    }
}

__global__ void equalize_and_reconstruct(unsigned char* d_image, int* d_cdf, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < width * height; i += stride) {
        int Y  = d_image[3*i + 0];
        int Cb = d_image[3*i + 1];
        int Cr = d_image[3*i + 2];

        int new_Y = d_cdf[Y];

        int R = min(255, max(0, (int)(new_Y + 1.402 * (Cr - 128))));
        int G = min(255, max(0, (int)(new_Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
        int B = min(255, max(0, (int)(new_Y + 1.772 * (Cb - 128))));

        d_image[3*i + 0] = R;
        d_image[3*i + 1] = G;
        d_image[3*i + 2] = B;
    }
}

int eq_GPU(unsigned char* image) {
    int image_size = width * height * pixelWidth;
    unsigned char* d_image;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    // Step 1: Convert to YCbCr on GPU
    rgb2ycbcr_kernel<<<numBlocks, blockSize>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy YCbCr image back to host
    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    // Step 2: CPU histogram and CDF
    int histogram[256] = {0};
    for (int i = 0; i < width * height * 3; i += 3) {
        int Y = image[i];
        histogram[Y]++;
    }

    int cdf[256] = {0};
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += histogram[i];
        cdf[i] = (int)((((float)sum - histogram[0]) / ((float)(width * height - 1))) * 255);
    }

    // Copy image and cdf back to device
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    int* d_cdf;
    cudaMalloc(&d_cdf, 256 * sizeof(int));
    cudaMemcpy(d_cdf, cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);

    // Step 3: Apply equalization + convert to RGB
    equalize_and_reconstruct<<<numBlocks, blockSize>>>(d_image, d_cdf, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_cdf);

    return 0;
}

int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_gpu.png";

    image = stbi_load(input, &width, &height, &pixelWidth, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return -1;
    }

    printf("Loaded image: %s (Width: %d, Height: %d, Channels: %d)\n", input, width, height, pixelWidth);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    printf("Processing on GPU...\n");
    eq_GPU(image);

    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros  = end.tv_usec - start.tv_usec;
    double elapsed_ms = seconds * 1000.0 + micros / 1000.0;
    printf(" Total GPU execution time: %.3f ms\n", elapsed_ms);

    printf("Saving result to: %s\n", output);
    stbi_write_png(output, width, height, pixelWidth, image, 0);
    stbi_image_free(image);

    return 0;
}
