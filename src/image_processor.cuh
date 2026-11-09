#ifndef IMAGE_PROCESSOR_CUH_
#define IMAGE_PROCESSOR_CUH_

#include <opencv2/core.hpp>

// Converts an 8-bit, 3-channel BGR image to grayscale using CUDA.
// Returns false if a CUDA operation fails.
bool ConvertToGrayscaleGpu(const cv::Mat& input, cv::Mat* output);

#endif  // IMAGE_PROCESSOR_CUH_
