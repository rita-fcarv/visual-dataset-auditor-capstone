#ifndef IMAGE_HASH_CUH_
#define IMAGE_HASH_CUH_

#include <cstdint>
#include <opencv2/core.hpp>

// Calculates a 64-bit difference hash from a grayscale image.
bool ComputeDHashGpu(const cv::Mat& grayscale, uint64_t* hash);

#endif  // IMAGE_HASH_CUH_
