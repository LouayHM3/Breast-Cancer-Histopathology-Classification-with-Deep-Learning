#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: oncology_benchmark <classifier.onnx> <image>\n";
    return EXIT_FAILURE;
  }
  try {
    const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("could not read image");
    }
    oncology::OnnxClassifier classifier(argv[1]);
    constexpr int warmup = 5;
    constexpr int iterations = 50;
    for (int index = 0; index < warmup; ++index) {
      classifier.classify(image);
    }
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (int index = 0; index < iterations; ++index) {
      const auto start = std::chrono::steady_clock::now();
      classifier.classify(image);
      const auto end = std::chrono::steady_clock::now();
      latencies.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
    const double mean =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / iterations;
    std::cout << std::fixed << std::setprecision(3)
              << "iterations=" << iterations << '\n'
              << "mean_latency_ms=" << mean << '\n'
              << "min_latency_ms="
              << *std::min_element(latencies.begin(), latencies.end()) << '\n'
              << "max_latency_ms="
              << *std::max_element(latencies.begin(), latencies.end()) << '\n'
              << "input_tensor_bytes=" << classifier.input_bytes() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
