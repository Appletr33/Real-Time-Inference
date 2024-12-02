/**
 * @file cuda_preprocess.h
 * @brief Header file for CUDA-based image preprocessing functions.
 *
 * This file contains functions for initializing, destroying, and running image preprocessing
 * using CUDA for accelerating operations like resizing and data format conversion.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

 /**
  * @brief Initialize CUDA resources for image preprocessing.
  *
  * Allocates resources and sets up the necessary environment for performing image preprocessing
  * on the GPU. This function should be called once before using any preprocessing functions.
  *
  * @param max_image_size The maximum image size (in pixels) that will be processed.
  */
void cuda_preprocess_init(int max_image_size);

/**
 * @brief Clean up and release CUDA resources.
 *
 * Frees any memory and resources allocated during initialization. This function should be
 * called when the preprocessing operations are no longer needed.
 */
void cuda_preprocess_destroy();

/**
 * @brief Preprocess an image using CUDA.
 *
 * This function resizes and converts the input image data (from uint8 to float) using CUDA
 * for faster processing. The result is stored in a destination buffer, ready for inference.
 *
 * @param src Pointer to the source image data in uint8 format.
 * @param src_width The width of the source image.
 * @param src_height The height of the source image.
 * @param dst Pointer to the destination buffer to store the preprocessed image in float format.
 * @param dst_width The desired width of the output image.
 * @param dst_height The desired height of the output image.
 * @param stream The CUDA stream to execute the preprocessing operation asynchronously.
 */

void cuda_preprocess_from_device(
    uint8_t* src_dev_ptr, // Source image data in device memory
    int src_width,        // Source image width
    int src_height,       // Source image height
    float* dst,           // Destination buffer in device memory
    int dst_width,        // Desired width for inference
    int dst_height,       // Desired height for inference
    cudaStream_t stream   // CUDA stream for asynchronous execution
);