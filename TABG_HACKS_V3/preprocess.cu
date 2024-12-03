// preprocessing.cu

#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

// Host and device pointers for image buffers (not needed anymore)
static uint8_t* img_buffer_host = nullptr;    // Pinned memory on the host for faster transfers
static uint8_t* img_buffer_device = nullptr;  // Memory on the device (GPU)

// Define the CUDA kernel for BGRA to RGB conversion and normalization
__global__ void bgra_to_rgb_kernel(cudaTextureObject_t tex, float* dst, int width, int height)
{
    // Calculate the x and y coordinates of the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Read BGRA from texture
    uchar4 bgra = tex2D<uchar4>(tex, x + 0.5f, y + 0.5f);

    // Convert BGRA to RGB and normalize to [0, 1]
    float r = bgra.z / 255.0f;
    float g = bgra.y / 255.0f;
    float b = bgra.x / 255.0f;

    // Calculate the index for each channel
    int idx = y * width + x;
    int area = width * height;

    // Write to separate channels (RGB)
    dst[idx] = r;            // Red channel
    dst[idx + area] = g;     // Green channel
    dst[idx + 2 * area] = b; // Blue channel
}

// Intermediary function to launch the kernel
void launch_bgra_to_rgb_kernel(cudaTextureObject_t texObj, float* dst, int width, int height, cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 block(16, 16); // 16x16 threads per block
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // Cover the entire image

    // Launch the CUDA kernel
    bgra_to_rgb_kernel<<<grid, block, 0, stream>>>(texObj, dst, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}