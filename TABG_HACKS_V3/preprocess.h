// preprocess.h

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <cuda_runtime.h>

// Declare the kernel
__global__ void bgra_to_rgb_kernel(cudaTextureObject_t tex, float* dst, int width, int height);

// Intermediary function declaration
void launch_bgra_to_rgb_kernel(cudaTextureObject_t texObj, float* dst, int width, int height, cudaStream_t stream);

#endif // PREPROCESS_H
