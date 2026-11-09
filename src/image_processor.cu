#include "image_processor.cuh"

#include <iostream>

#include <cuda_runtime.h>

namespace {

__global__ void GrayscaleKernel(const unsigned char* input,
                                unsigned char* output,
                                int num_pixels) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_pixels) {
    return;
  }

  unsigned char blue = input[3 * index];
  unsigned char green = input[3 * index + 1];
  unsigned char red = input[3 * index + 2];

  output[index] = static_cast<unsigned char>(
      0.114f * blue + 0.587f * green + 0.299f * red);
}

bool CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::cerr << operation << ": "
              << cudaGetErrorString(error) << "\n";
    return false;
  }
  return true;
}

}  // namespace

bool ConvertToGrayscaleGpu(const cv::Mat& input, cv::Mat* output) {
  if (output == nullptr || input.empty() ||
      input.type() != CV_8UC3) {
    return false;
  }

  cv::Mat contiguous_input = input.isContinuous()
                                 ? input
                                 : input.clone();

  const size_t num_pixels = contiguous_input.total();
  const size_t input_bytes = num_pixels * 3;
  const size_t output_bytes = num_pixels;

  unsigned char* device_input = nullptr;
  unsigned char* device_output = nullptr;
  bool success = false;

  do {
    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_input),
            input_bytes), "cudaMalloc input")) {
      break;
    }

    if (!CheckCuda(cudaMalloc(
            reinterpret_cast<void**>(&device_output),
            output_bytes), "cudaMalloc output")) {
      break;
    }

    if (!CheckCuda(cudaMemcpy(
            device_input, contiguous_input.data, input_bytes,
            cudaMemcpyHostToDevice), "Copy input")) {
      break;
    }

    const int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (num_pixels + threads_per_block - 1) / threads_per_block);

    GrayscaleKernel<<<blocks, threads_per_block>>>(
        device_input, device_output,
        static_cast<int>(num_pixels));

    if (!CheckCuda(cudaGetLastError(), "Kernel launch") ||
        !CheckCuda(cudaDeviceSynchronize(), "Kernel execution")) {
      break;
    }

    output->create(input.rows, input.cols, CV_8UC1);

    if (!CheckCuda(cudaMemcpy(
            output->data, device_output, output_bytes,
            cudaMemcpyDeviceToHost), "Copy output")) {
      break;
    }

    success = true;
  } while (false);

  if (device_input != nullptr) {
    CheckCuda(cudaFree(device_input), "Free input");
  }
  if (device_output != nullptr) {
    CheckCuda(cudaFree(device_output), "Free output");
  }

  return success;
}
