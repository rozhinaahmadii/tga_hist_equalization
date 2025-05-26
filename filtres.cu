// Inclourem totes les biblioteques necessàries
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

using namespace cv;
using namespace std;

#define CHANNELS 3
#define HISTOGRAM_SIZE 256

// ============================= KERNELS CUDA =============================

// Filtre gaussià 3x3 (en escala de grisos, canal Y)
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float kernel[3][3] = {
        {1/16.f, 2/16.f, 1/16.f},
        {2/16.f, 4/16.f, 2/16.f},
        {1/16.f, 2/16.f, 1/16.f}
    };

    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            sum += input[ny * width + nx] * kernel[ky + 1][kx + 1];
        }
    }
    output[y * width + x] = (unsigned char)sum;
}

// Histograma amb memòria compartida
__global__ void computeHistogramShared(const unsigned char* input, int* histogram, int width, int height) {
    __shared__ unsigned int local_hist[HISTOGRAM_SIZE];
    int tid = threadIdx.x;

    // Inicialitzem histograma local
    if (tid < HISTOGRAM_SIZE)
        local_hist[tid] = 0;
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < width * height; i += stride) {
        atomicAdd(&(local_hist[input[i]]), 1);
    }
    __syncthreads();

    // Acumulem a histograma global
    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&(histogram[tid]), local_hist[tid]);
    }
}

// Càlcul CDF i equalització
__global__ void applyHistogramEqualization(unsigned char* image, const int* cdf, int width, int height, int cdf_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < width * height; i += stride) {
        unsigned char val = image[i];
        int eq_val = ((cdf[val] - cdf_min) * 255) / (width * height - cdf_min);
        image[i] = min(max(eq_val, 0), 255);
    }
}

// =========================== FUNCIONS AUXILIARS ============================

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " -> " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// =============================== MAIN ===================================

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./program <image_path>\n";
        return -1;
    }

    auto start_time = chrono::high_resolution_clock::now();

    // Carreguem la imatge i la convertim a YCbCr
    Mat input_image = imread(argv[1], IMREAD_COLOR);
    if (input_image.empty()) {
        cerr << "No s'ha pogut carregar la imatge." << endl;
        return -1;
    }

    Mat ycbcr_image;
    cvtColor(input_image, ycbcr_image, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycbcr_image, channels);

    int width = input_image.cols;
    int height = input_image.rows;
    size_t image_size = width * height;

    // Alliberem memòria host i device
    unsigned char *d_y_in, *d_y_blurred;
    int *d_histogram;

    checkCuda(cudaMalloc(&d_y_in, image_size), "cudaMalloc d_y_in");
    checkCuda(cudaMalloc(&d_y_blurred, image_size), "cudaMalloc d_y_blurred");
    checkCuda(cudaMemcpy(d_y_in, channels[0].data, image_size, cudaMemcpyHostToDevice), "cudaMemcpy Y input");

    // Filtrem el canal Y (gaussian blur)
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    gaussianBlurKernel<<<grid, block>>>(d_y_in, d_y_blurred, width, height);

    // Calcular histograma amb memòria compartida
    checkCuda(cudaMalloc(&d_histogram, HISTOGRAM_SIZE * sizeof(int)), "cudaMalloc histograma");
    checkCuda(cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(int)), "cudaMemset histograma");
    computeHistogramShared<<<32, 256>>>(d_y_blurred, d_histogram, width, height);

    // Transferim a host per calcular CDF
    int h_histogram[HISTOGRAM_SIZE];
    checkCuda(cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost), "copy histograma to host");

    int cdf[HISTOGRAM_SIZE] = {0};
    cdf[0] = h_histogram[0];
    int cdf_min = 0;
    for (int i = 1; i < HISTOGRAM_SIZE; ++i) {
        cdf[i] = cdf[i - 1] + h_histogram[i];
        if (cdf_min == 0 && cdf[i] != 0) cdf_min = cdf[i];
    }

    // Copiem CDF a device
    int* d_cdf;
    checkCuda(cudaMalloc(&d_cdf, HISTOGRAM_SIZE * sizeof(int)), "cudaMalloc d_cdf");
    checkCuda(cudaMemcpy(d_cdf, cdf, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_cdf");

    // Equalitzem
    applyHistogramEqualization<<<32, 256>>>(d_y_blurred, d_cdf, width, height, cdf_min);
    checkCuda(cudaMemcpy(channels[0].data, d_y_blurred, image_size, cudaMemcpyDeviceToHost), "copy back equalized Y");

    // Reconstruïm la imatge
    merge(channels, ycbcr_image);
    Mat output_image;
    cvtColor(ycbcr_image, output_image, COLOR_YCrCb2BGR);

    imwrite("output_equalized_gpu.png", output_image);
    imwrite("output_ycbcr_gpu.png", ycbcr_image);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;
    cout << "✅ GPU histogram equalization with filter done in " << duration.count() << " ms." << endl;

    // Free
    cudaFree(d_y_in);
    cudaFree(d_y_blurred);
    cudaFree(d_histogram);
    cudaFree(d_cdf);

    return 0;
}
