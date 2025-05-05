# CUDA Histogram Equalization 
This project implements a high-performance histogram equalization algorithm for **color images** using **NVIDIA CUDA**. The goal is to enhance contrast in images with low definition or excessive brightness, especially in contexts like medical imaging and visual enhancement. This project is developed as part of the *Tarjetas Gráficas y Aceleradores* (TGA) course at UPC.

---

## ✅ Key Features

- Histogram equalization on color images using **RGB or YUV** approaches
- CUDA implementation on **1 GPU**, supporting:
  - Block, row, and column kernel variants
  - Shared memory optimizations
  - Pinned memory usage
  - Multiple CUDA streams for pipelining
- **Multi-GPU support** for 1, 2, and 4 GPUs
- Benchmarking tools with kernel execution timing and speedup analysis
- Modular structure for easy testing and future extensions

---

## 📌 Project Structure and Phases

1. **CPU baseline version** to understand the algorithm
2. **CUDA-based GPU implementation** with various kernel strategies
3. **Memory optimization techniques**: shared memory, pinned memory, streams
4. **Multi-GPU implementation** (1, 2, and 4 GPUs)
5. **Benchmarking and comparison** with the CPU version

We will also present:
- Tables with execution times per kernel
- Speedup graphs comparing 1, 2, and 4 GPUs
- Visual comparisons of original and equalized images

---

## 🚀 Getting Started

### Prerequisites

- CUDA-compatible GPU
- CUDA Toolkit installed
- C++ compiler (e.g., `g++`)
- Optional: OpenCV or stb_image for image I/O

### Build and Run

> Instructions and Makefile will be added once the implementation is complete.

job.sh: Submits a CUDA job to a GPU queue.

---

## 📊 Evaluation & Benchmarking

We will:
- Compare performance across kernel variants
- Measure speedup vs. the CPU version
- Analyze scaling across multiple GPUs
- Visualize results with graphs and enhanced images

---

## 🧪 Sample Test Images

We'll test with images of various types:
- Grayscale and full-color (RGB)
- Low-contrast and high-brightness images
- Resolutions from 256×256 to 4K and above

---

## 📚 References

- Course: *Tarjetas Gráficas y Aceleradores* (TGA), UPC 2024–25
- CUDA Programming Guide – NVIDIA
- OpenCV Documentation

---

## 👩‍💻 Authors

- Rozhina Ahmadi  
- Jan

---
