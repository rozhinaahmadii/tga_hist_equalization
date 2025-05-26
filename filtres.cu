##Afegim un filtre simple de blur (3x3 box filter)

##Afegim aquest kernel abans del càlcul del histograma. Farem blur només sobre el canal Y,
##ja que és el que s'usa per a l’histograma.

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
                sum += d_image[idx + 0];  // només canal Y
            }
        }
        int out_idx = (row * width + col) * 3;
        d_blurred[out_idx + 0] = (unsigned char)(sum / 9.0f);
        d_blurred[out_idx + 1] = d_image[out_idx + 1]; // copia Cb
        d_blurred[out_idx + 2] = d_image[out_idx + 2]; // copia Cr
    }
}

##Histograma amb memòria compartida

##Aquest kernel utilitza __shared__ per fer un histograma local per bloc, i després el combina globalment:

__global__ void histogram_shared(unsigned char* d_image, unsigned int* d_hist, int width, int height) {
    __shared__ unsigned int local_hist[256];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Inicialització de l'histograma compartit
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

    // Reducció: suma local → global
    for (int i = tid; i < 256; i += blockDim.x * blockDim.y)
        atomicAdd(&d_hist[i], local_hist[i]);
}
 ##Modificacions a eq_GPU

##Ara actualitzem el flux en eq_GPU:

int eq_GPU(unsigned char* image) {
    int image_size = width * height * pixelWidth;
    unsigned char *d_image, *d_blurred;
    unsigned int* d_hist;

    cudaMalloc(&d_image, image_size);
    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);

    // Reserva per la imatge amb blur
    cudaMalloc(&d_blurred, image_size);

    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 1. Convertir RGB a YCbCr i calcular histograma amb shared memory
    rgb2ycbcr_rowwise<<<grid, block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    // 2. Aplicar blur (només canal Y)
    blur_Y_channel<<<grid, block>>>(d_image, d_blurred, width, height);
    cudaDeviceSynchronize();

    // Substituïm la imatge original per la imatge amb blur
    cudaMemcpy(d_image, d_blurred, image_size, cudaMemcpyDeviceToDevice);

    // 3. Calcular histograma amb shared memory (usant imatge amb blur)
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    histogram_shared<<<grid, block>>>(d_image, d_hist, width, height);
    cudaDeviceSynchronize();

    // 4. Passar a host
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

    // 5. Aplicar igualació i reconstruir
    equalize_and_reconstruct_rowwise<<<grid, block>>>(d_image, d_cdf, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_blurred);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}
