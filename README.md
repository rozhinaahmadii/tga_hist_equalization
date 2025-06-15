# CUDA Histogram Equalization

## Authors
- Rozhina Ahmadi  
- Jan Antón Villanueva

## Course
**Tarjetas Gráficas y Aceleradores (2024-25 Q2)**  
Department of Computer Architecture

---

## 📝 Project Summary

Histogram Equalization is a common technique used to enhance image contrast, especially for low-light or low-contrast conditions. Applications include medical imaging, object detection, and photography.

In this project, we implement and analyze histogram equalization for 24-bit RGB images using CUDA. We compare the CPU baseline with various GPU implementations, including single and multi-GPU strategies.

---

## ⚙️ Algorithm Overview

The equalization process includes the following steps:

1. **RGB to YCbCr** conversion (only the Y channel is equalized)
2. **Histogram calculation** of the Y channel
3. **CDF (Cumulative Distribution Function)** computation
4. **Equalization mapping** using the CDF
5. **RGB image reconstruction**

---

## 🚀 Implemented Versions

| ID | Type      | Description                          | Memory      | Streams | Filename                             |
|----|-----------|--------------------------------------|-------------|---------|--------------------------------------|
| 1  | CPU       | Baseline                             | Host        | No      | `eq_CPU.cu`                          |
| 2  | 1 GPU     | Row-wise, global memory              | Global      | No      | `eq_rowwise_global.cu`              |
| 3  | 1 GPU     | Row-wise, shared memory (standard)   | Shared Std  | No      | `eq_rowwise_shared_standard.cu`     |
| 4  | 1 GPU     | Row-wise, shared memory (pinned)     | Shared Pinned | No    | `eq_rowwise_shared_pinned.cu`       |
| 5  | 1 GPU     | Shared pinned + CUDA streams         | Shared Pinned | Yes   | `eq_rowwise_shared_pinned_streams.cu` |
| 6  | 1 GPU     | Shared standard + CUDA streams       | Shared Std  | Yes     | `eq_rowwise_shared_standard_streams.cu` |
| 7  | 2 GPUs    | Multi-GPU version (no streams)       | Shared Pinned | No    | `2GPUs.cu`                     |
| 8  | 4 GPUs    | Multi-GPU version (no streams)       | Shared Pinned | No    | `4GPUs.cu`                     |

---

## 📊 Performance Summary

### Image IMG01 (5184 × 3456 pixels)

| ID | Type    | Total Kernel Time (ms) | Speedup (vs CPU) |
|----|---------|------------------------|------------------|
| 1  | CPU     | 207.444                | 1×               |
| 2  | 1 GPU   | 11.576                 | 17.92×           |
| 3  | 1 GPU   | 6.912                  | 30.01×           |
| 4  | 1 GPU   | 6.871                  | 30.19×           |
| 5  | 1 GPU   | 3.481                  | 59.59×           |
| 6  | 1 GPU   | 3.433                  | 60.43×           |
| 8  | 4 GPUs  | 16.094                 | 0.43× (slower)   |

### Image IMG00 (1366 × 876 pixels)

| ID | Type    | Total Kernel Time (ms) | Speedup (vs CPU) |
|----|---------|------------------------|------------------|
| 1  | CPU     | 35.022                 | 1×               |
| 4  | 1 GPU   | 0.800                  | 43.78×           |
| 7  | 2 GPUs  | 2.489                  | 0.32× (slower)   |
| 8  | 4 GPUs  | 2.267                  | 0.35× (slower)   |

> ⚠️ Multi-GPU versions performed worse than single GPU due to overhead, lack of overlapping transfers, and coordination complexity.

---

## 📦 How to Run

### 1. Compile with `make` and run the version
```bash
make
sbatch job.sh
```
## ✅ Expected Results
You can expect a 40–60x speedup on large images with the optimized single-GPU versions using shared memory and streams.
The GPU-accelerated versions exclude I/O time in performance metrics. All kernel timings were measured using cudaEventElapsedTime().

