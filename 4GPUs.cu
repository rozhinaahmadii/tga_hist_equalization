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

// Kernels (sin cambios)
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

// Función principal modificada para 4 GPUs
int eq_GPU_multi(unsigned char* image) {
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    const int num_gpus = 4;
    
    // Inicializar todas las GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
    }

    // Dividir la imagen en 4 partes aproximadamente iguales
    int part_height[num_gpus];
    int remaining = height;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        part_height[gpu] = (remaining + (num_gpus - gpu - 1)) / (num_gpus - gpu);
        remaining -= part_height[gpu];
    }

    int part_size[num_gpus];
    unsigned char* h_image_part[num_gpus];
    unsigned char* d_image[num_gpus], *d_blurred[num_gpus];
    unsigned int* d_hist[num_gpus];
    int* d_cdf[num_gpus];

    // Calcular tamaños y punteros para cada parte
    h_image_part[0] = image;
    part_size[0] = part_height[0] * width * pixelWidth;
    
    for (int gpu = 1; gpu < num_gpus; gpu++) {
        part_size[gpu] = part_height[gpu] * width * pixelWidth;
        h_image_part[gpu] = h_image_part[gpu-1] + part_size[gpu-1];
    }

    // Asignar memoria en cada GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc((void**)&d_image[gpu], part_size[gpu]);
        cudaMalloc((void**)&d_blurred[gpu], part_size[gpu]);
        cudaMalloc((void**)&d_hist[gpu], 256 * sizeof(unsigned int));
        cudaMalloc((void**)&d_cdf[gpu], 256 * sizeof(int));
    }

    // Copiar datos a cada GPU (asíncrono con streams)
    cudaStream_t streams[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamCreate(&streams[gpu]);
        cudaMemcpyAsync(d_image[gpu], h_image_part[gpu], part_size[gpu], 
                       cudaMemcpyHostToDevice, streams[gpu]);
    }

    // Configurar kernels
    dim3 block(32, 32);

    // Ejecutar kernels en paralelo en todas las GPUs
    cudaEvent_t events[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaEventCreate(&events[gpu]);
        
        dim3 grid((width + block.x - 1) / block.x, (part_height[gpu] + block.y - 1) / block.y);

        // Paso 1: RGB → YCbCr + histograma inicial
        rgb2ycbcr_rowwise<<<grid, block, 0, streams[gpu]>>>(d_image[gpu], d_hist[gpu], width, part_height[gpu]);
        
        // Paso 2: Blur Y channel
        blur_Y_channel<<<grid, block, 0, streams[gpu]>>>(d_image[gpu], d_blurred[gpu], width, part_height[gpu]);
        cudaMemcpyAsync(d_image[gpu], d_blurred[gpu], part_size[gpu], 
                       cudaMemcpyDeviceToDevice, streams[gpu]);
        
        // Paso 3: Histograma con memoria compartida
        cudaMemsetAsync(d_hist[gpu], 0, 256 * sizeof(unsigned int), streams[gpu]);
        histogram_shared<<<grid, block, 0, streams[gpu]>>>(d_image[gpu], d_hist[gpu], width, part_height[gpu]);
        
        cudaEventRecord(events[gpu], streams[gpu]);
    }

    // Esperar que todas las GPUs terminen
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaEventSynchronize(events[gpu]);
    }

    // Combinar histogramas parciales en CPU
    unsigned int h_hist_combined[256] = {0};
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        unsigned int h_hist_partial[256];
        cudaSetDevice(gpu);
        cudaMemcpy(h_hist_partial, d_hist[gpu], 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 256; i++) h_hist_combined[i] += h_hist_partial[i];
    }

    // Calcular CDF global
    int h_cdf[256] = {0}, sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += h_hist_combined[i];
        h_cdf[i] = (int)(((float)sum - h_hist_combined[0]) / (width * height - 1) * 255);
    }

    // Copiar CDF a todas las GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaMemcpy(d_cdf[gpu], h_cdf, 256 * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Aplicar equalización y reconstrucción
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        dim3 grid((width + block.x - 1) / block.x, (part_height[gpu] + block.y - 1) / block.y);
        equalize_and_reconstruct_rowwise<<<grid, block, 0, streams[gpu]>>>(d_image[gpu], d_cdf[gpu], width, part_height[gpu]);
        cudaMemcpyAsync(h_image_part[gpu], d_image[gpu], part_size[gpu], 
                       cudaMemcpyDeviceToHost, streams[gpu]);
    }

    // Sincronizar todas las GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamSynchronize(streams[gpu]);
    }

    // Liberar memoria
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_image[gpu]);
        cudaFree(d_blurred[gpu]);
        cudaFree(d_hist[gpu]);
        cudaFree(d_cdf[gpu]);
        cudaStreamDestroy(streams[gpu]);
        cudaEventDestroy(events[gpu]);
    }

    gettimeofday(&end_total, NULL);
    double total_time = (end_total.tv_sec - start_total.tv_sec) * 1000.0 + 
                       (end_total.tv_usec - start_total.tv_usec) / 1000.0;
    printf("\n✅ Total GPU (%d GPUs) time: %.3f ms\n", num_gpus, total_time);

    return 0;
}

// Función main (sin cambios necesarios)
int main(int argc, char** argv) {
    const char* input = "./IMG/IMG00.jpg";
    const char* output = "output_equalized_4gpu.png";

    int n_channels;
    unsigned char* raw = stbi_load(input, &width, &height, &n_channels, 0);
    if (!raw) {
        fprintf(stderr, "❌ Couldn't load image.\n");
        return -1;
    }

    pixelWidth = n_channels;
    int size = width * height * pixelWidth;

    // Allocate pinned memory
    cudaHostAlloc((void**)&image, size, cudaHostAllocDefault);
    memcpy(image, raw, size);
    stbi_image_free(raw);

    printf("📷 Image loaded: %d x %d (Channels: %d)\n", width, height, pixelWidth);

    // Verificar número de GPUs disponibles
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    // Ejecutar en 4 GPUs
    struct timeval start, end;
    gettimeofday(&start, NULL);
    eq_GPU_multi(image);
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + 
                    (end.tv_usec - start.tv_usec) / 1000.0;
    printf("🕒 Total runtime (incl. GPU): %.3f ms\n", elapsed);

    // Guardar imagen
    stbi_write_png(output, width, height, pixelWidth, image, 0);
    cudaFreeHost(image);

    return 0;
}
