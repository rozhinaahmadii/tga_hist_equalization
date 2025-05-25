#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Kernel per convertir RGB a YCbCr i generar l’histograma
__global__ void rgb2ycbcr_rowwise(uchar3* image, int* hist, int width, int height) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        uchar3 pixel = image[idx];

        float r = pixel.x;
        float g = pixel.y;
        float b = pixel.z;

        unsigned char Y = 0.299f * r + 0.587f * g + 0.114f * b;
        unsigned char Cb = -0.1687f * r - 0.3313f * g + 0.5f * b + 128;
        unsigned char Cr = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

        image[idx] = make_uchar3(Y, Cb, Cr);
        atomicAdd(&hist[Y], 1);
    }
}

// Kernel per calcular el CDF prefixat a GPU
__global__ void compute_cdf_gpu(int* hist, float* cdf, int total_pixels) {
    __shared__ int temp[256];
    int t = threadIdx.x;

    if (t < 256) temp[t] = hist[t];
    __syncthreads();

    for (int offset = 1; offset < 256; offset *= 2) {
        int val = 0;
        if (t >= offset) val = temp[t - offset];
        __syncthreads();
        temp[t] += val;
        __syncthreads();
    }

    if (t < 256) {
        cdf[t] = (float)temp[t] / total_pixels;
        if (t % 64 == 0) {
            printf("CDF[%d] = %.4f\n", t, cdf[t]);
        }
    }
}

// Kernel per aplicar l’equalització i tornar a RGB
__global__ void equalize_and_reconstruct_rowwise(uchar3* image, float* cdf, int width, int height) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        uchar3 ycbcr = image[idx];

        unsigned char Y = ycbcr.x;
        unsigned char Cb = ycbcr.y;
        unsigned char Cr = ycbcr.z;

        float newY = 255.0f * cdf[Y];

        int r = newY + 1.402f * (Cr - 128);
        int g = newY - 0.3441f * (Cb - 128) - 0.7141f * (Cr - 128);
        int b = newY + 1.772f * (Cb - 128);

        image[idx] = make_uchar3(
            min(max(r, 0), 255),
            min(max(g, 0), 255),
            min(max(b, 0), 255)
        );
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Ús: %s <imatge_entrada>\n", argv[0]);
        return -1;
    }

    // Carregar la imatge original
    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        printf("No s'ha pogut carregar la imatge.\n");
        return -1;
    }

    // Guardar l'original
    imwrite("output_original.png", image);

    int width = image.cols;
    int height = image.rows;
    int total_pixels = width * height;
    size_t size = total_pixels * sizeof(uchar3);

    // Reservar memòria a GPU
    uchar3* d_image;
    int* d_hist;
    float* d_cdf;

    cudaMalloc(&d_image, size);
    cudaMemcpy(d_image, image.ptr<uchar3>(), size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    cudaMalloc(&d_cdf, 256 * sizeof(float));

    // Inici cronometratge
    auto start = high_resolution_clock::now();

    // Pas 1: conversió + histograma
    rgb2ycbcr_rowwise<<<height, width>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    // Pas 2: CDF a GPU
    compute_cdf_gpu<<<1, 256>>>(d_hist, d_cdf, total_pixels);
    cudaDeviceSynchronize();

    // Depuració: copiar histograma i CDF a CPU
    int h_hist[256];
    float h_cdf[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cdf, d_cdf, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    int zeros = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_hist[i] == 0) zeros++;
        if (i % 64 == 0) {
            printf("hist[%d] = %d\tcdf[%d] = %.4f\n", i, h_hist[i], i, h_cdf[i]);
        }
    }
    if (zeros >= 255) {
        printf("⚠️ AVÍS: la imatge té molt poc contrast! No s’esperen canvis visibles.\n");
    }

    // Pas 3: Equalització i reconstrucció
    equalize_and_reconstruct_rowwise<<<height, width>>>(d_image, d_cdf, width, height);
    cudaDeviceSynchronize();

    // Fi cronometratge
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("⏱️ Temps d'execució GPU: %lld ms\n", duration.count());

    // Copiar resultat a CPU i guardar
    cudaMemcpy(image.ptr<uchar3>(), d_image, size, cudaMemcpyDeviceToHost);
    imwrite("output_equalized_gpu.png", image);

    // Alliberar
    cudaFree(d_image);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}
