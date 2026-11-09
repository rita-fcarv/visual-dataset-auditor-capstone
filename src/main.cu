#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/imgcodecs.hpp>

#include "image_hash.cuh"
#include "image_metrics.cuh"
#include "image_processor.cuh"
#include "similarity.cuh"

namespace {

bool IsImageFile(const std::string& filename) {
  const size_t dot = filename.find_last_of('.');

  if (dot == std::string::npos) {
    return false;
  }

  std::string extension = filename.substr(dot);

  for (char& character : extension) {
    if (character >= 'A' && character <= 'Z') {
      character = character - 'A' + 'a';
    }
  }

  return extension == ".png" ||
         extension == ".jpg" ||
         extension == ".jpeg" ||
         extension == ".bmp" ||
         extension == ".ppm";
}

std::vector<std::string> ListImages(
    const std::string& directory) {
  std::vector<std::string> images;

  DIR* dir = opendir(directory.c_str());

  if (dir == nullptr) {
    std::cerr << "Cannot open directory: "
              << directory << "\n";
    return images;
  }

  struct dirent* entry;

  while ((entry = readdir(dir)) != nullptr) {
    const std::string filename = entry->d_name;

    if (IsImageFile(filename)) {
      const std::string path = directory + "/" + filename;
      struct stat file_info;

      if (stat(path.c_str(), &file_info) == 0 &&
          S_ISREG(file_info.st_mode)) {
        images.push_back(filename);
      }
    }
  }

  closedir(dir);

  std::sort(images.begin(), images.end());

  return images;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string input_directory;
  std::string output_directory;
  int similarity_threshold = 5;

  // Parse command-line arguments.
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if (argument == "--input" && i + 1 < argc) {
      input_directory = argv[++i];

    } else if (argument == "--output" && i + 1 < argc) {
      output_directory = argv[++i];

    } else if (argument == "--threshold" && i + 1 < argc) {
      try {
        const std::string value = argv[++i];
        size_t parsed = 0;
        similarity_threshold = std::stoi(value, &parsed);

        if (parsed != value.size() ||
            similarity_threshold < 0 ||
            similarity_threshold > 64) {
          std::cerr << "Threshold must be between 0 and 64.\n";
          return 1;
        }
      } catch (const std::exception&) {
        std::cerr << "Invalid similarity threshold.\n";
        return 1;
      }

    } else {
      std::cerr << "Unknown or incomplete argument: "
                << argument << "\n";
      return 1;
    }
  }

  if (input_directory.empty() || output_directory.empty()) {
    std::cerr
        << "Usage: " << argv[0]
        << " --input <directory>"
        << " --output <directory>"
        << " [--threshold 0-64]\n";
    return 1;
  }

  // Validate output directory.
  struct stat directory_info;

  if (stat(output_directory.c_str(), &directory_info) != 0 ||
      !S_ISDIR(directory_info.st_mode)) {
    std::cerr << "Output directory must already exist: "
              << output_directory << "\n";
    return 1;
  }

  // Discover input images.
  const std::vector<std::string> images =
      ListImages(input_directory);

  if (images.empty()) {
    std::cerr << "No images found.\n";
    return 1;
  }

  // Open quality report.
  std::ofstream quality_report(
      output_directory + "/quality_report.csv");

  if (!quality_report.is_open()) {
    std::cerr << "Could not create quality report.\n";
    return 1;
  }

  quality_report
      << "filename,mean_brightness,contrast,blur_score\n";

  quality_report << std::fixed << std::setprecision(4);

  std::vector<uint64_t> image_hashes;
  std::vector<std::string> processed_filenames;

  image_hashes.reserve(images.size());
  processed_filenames.reserve(images.size());

  int processed = 0;
  int failed = 0;

  const auto start = std::chrono::steady_clock::now();

  // Process all images.
  for (const std::string& filename : images) {
    const std::string input_path =
        input_directory + "/" + filename;

    const cv::Mat image =
        cv::imread(input_path, cv::IMREAD_COLOR);

    if (image.empty()) {
      ++failed;
      std::cerr << "Failed to read: " << filename << "\n";
      continue;
    }

    // CUDA grayscale conversion.
    cv::Mat grayscale;

    if (!ConvertToGrayscaleGpu(image, &grayscale)) {
      ++failed;
      std::cerr << "GPU processing failed: "
                << filename << "\n";
      continue;
    }

    // CUDA image quality metrics.
    ImageMetrics metrics;

    if (!ComputeImageMetricsGpu(grayscale, &metrics)) {
      ++failed;
      std::cerr << "Quality analysis failed: "
                << filename << "\n";
      continue;
    }

    // CUDA perceptual hashing.
    uint64_t image_hash = 0;

    if (!ComputeDHashGpu(grayscale, &image_hash)) {
      ++failed;
      std::cerr << "Hash calculation failed: "
                << filename << "\n";
      continue;
    }

    // Save grayscale output.
    const size_t dot = filename.find_last_of('.');
    const std::string base = filename.substr(0, dot);

    const std::string output_path =
        output_directory + "/" + base + ".png";

    bool saved = false;

    try {
      saved = cv::imwrite(output_path, grayscale);
    } catch (const cv::Exception& error) {
      std::cerr << "OpenCV write error: "
                << error.what() << "\n";
    }

    if (!saved) {
      ++failed;
      std::cerr << "Failed to save: "
                << output_path << "\n";
      continue;
    }

    // Record image quality metrics.
    quality_report
        << filename << ","
        << metrics.mean_brightness << ","
        << metrics.contrast << ","
        << metrics.blur_score << "\n";

    // Store hashes for similarity comparison.
    image_hashes.push_back(image_hash);
    processed_filenames.push_back(filename);

    ++processed;
  }

  quality_report.close();

  // Compare image hashes using CUDA.
  std::vector<SimilarPair> similar_pairs;

  if (!FindSimilarPairsGpu(
          image_hashes,
          similarity_threshold,
          &similar_pairs)) {
    std::cerr << "GPU similarity comparison failed.\n";
    return 1;
  }

  // Generate similarity report.
  std::ofstream similarity_report(
      output_directory + "/similarity_report.csv");

  if (!similarity_report.is_open()) {
    std::cerr << "Could not create similarity report.\n";
    return 1;
  }

  similarity_report
      << "image_1,image_2,hamming_distance\n";

  for (const SimilarPair& pair : similar_pairs) {
    similarity_report
        << processed_filenames[pair.first_index] << ","
        << processed_filenames[pair.second_index] << ","
        << pair.distance << "\n";
  }

  similarity_report.close();

  const auto end = std::chrono::steady_clock::now();

  const double seconds =
      std::chrono::duration<double>(end - start).count();

  // Execution summary.
  std::cout << "\n=== Visual Dataset Auditor ===\n";
  std::cout << "Images found:        "
            << images.size() << "\n";
  std::cout << "Images processed:    "
            << processed << "\n";
  std::cout << "Images failed:       "
            << failed << "\n";
  std::cout << "Similar image pairs: "
            << similar_pairs.size() << "\n";
  std::cout << "Similarity threshold: "
            << similarity_threshold << "\n";
  std::cout << "Elapsed time:        "
            << seconds << " s\n";

  std::cout << "\nReports saved to: "
            << output_directory << "\n";

  return failed == 0 ? 0 : 1;
}
