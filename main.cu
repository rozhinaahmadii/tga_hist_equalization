#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./main.exe input.jpg output.jpg\n";
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    int width, height, channels;
    unsigned char* img = stbi_load(input_path, &width, &height, &channels, 0);

    if (!img) {
        std::cerr << "Error: could not load image " << input_path << "\n";
        return 1;
    }

    std::cout << "Loaded image: " << input_path << " (" << width << "x" << height << "), channels: " << channels << "\n";

    // For now, convert to grayscale as a placeholder step
    unsigned char* gray = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++) {
        int r = img[i * channels + 0];
        int g = img[i * channels + 1];
        int b = img[i * channels + 2];
        gray[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }

    // Save grayscale image as PNG (or JPG)
    if (!stbi_write_png(output_path, width, height, 1, gray, width)) {
        std::cerr << "Error: could not save output to " << output_path << "\n";
        return 1;
    }

    std::cout << "Saved grayscale image to " << output_path << "\n";

    stbi_image_free(img);
    delete[] gray;

    return 0;
}
