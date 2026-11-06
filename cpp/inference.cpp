#include "inference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

namespace oncology {

OnnxClassifier::OnnxClassifier(const std::string& model_path, int image_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "oncology"),
      image_size_(image_size) {
  if (image_size_ <= 0) {
    throw std::invalid_argument("image_size must be positive");
  }
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetIntraOpNumThreads(1);
  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  auto input_name = session_.GetInputNameAllocated(0, allocator_);
  auto output_name = session_.GetOutputNameAllocated(0, allocator_);
  if (!input_name || !output_name) {
    throw std::runtime_error("ONNX model has no input or output name");
  }
  input_name_ = input_name.get();
  output_name_ = output_name.get();
}

std::vector<float> OnnxClassifier::preprocess(const cv::Mat& image) const {
  if (image.empty()) {
    throw std::invalid_argument("image is empty");
  }
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else {
    throw std::invalid_argument("image must have one or three channels");
  }
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(image_size_, image_size_));
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

  constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
  constexpr std::array<float, 3> stddev{0.229F, 0.224F, 0.225F};
  std::vector<float> tensor(static_cast<std::size_t>(3) * image_size_ * image_size_);
  for (int row = 0; row < image_size_; ++row) {
    for (int col = 0; col < image_size_; ++col) {
      const auto pixel = resized.at<cv::Vec3f>(row, col);
      const auto offset = static_cast<std::size_t>(row) * image_size_ + col;
      for (int channel = 0; channel < 3; ++channel) {
        tensor[static_cast<std::size_t>(channel) * image_size_ * image_size_ + offset] =
            (pixel[channel] - mean[channel]) / stddev[channel];
      }
    }
  }
  return tensor;
}

Prediction OnnxClassifier::postprocess(const std::vector<float>& logits,
                                       double preprocess_ms,
                                       double inference_ms) const {
  if (logits.size() < 2) {
    throw std::runtime_error("ONNX output must contain two class logits");
  }
  const float maximum = std::max(logits[0], logits[1]);
  const float benign_exp = std::exp(logits[0] - maximum);
  const float malignant_exp = std::exp(logits[1] - maximum);
  const float denominator = benign_exp + malignant_exp;
  const float benign_probability = benign_exp / denominator;
  const float malignant_probability = malignant_exp / denominator;
  const bool malignant = malignant_probability >= benign_probability;
  return {
      malignant ? "malignant" : "benign",
      malignant ? malignant_probability : benign_probability,
      benign_probability,
      malignant_probability,
      preprocess_ms,
      inference_ms,
      0.0,
  };
}

Prediction OnnxClassifier::classify(const cv::Mat& image) const {
  const auto preprocess_start = Clock::now();
  const auto values = preprocess(image);
  const auto preprocess_end = Clock::now();
  const std::array<int64_t, 4> shape{1, 3, image_size_, image_size_};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(values.data()), values.size(),
      shape.data(), shape.size());
  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  const auto inference_start = Clock::now();
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                              output_names, 1);
  const auto inference_end = Clock::now();
  if (outputs.empty() || !outputs[0].IsTensor()) {
    throw std::runtime_error("ONNX model returned no tensor output");
  }
  const auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto logits = outputs[0].GetTensorData<float>();
  const auto count = tensor_info.GetElementCount();
  std::vector<float> values_out(logits, logits + count);
  const auto postprocess_start = Clock::now();
  auto prediction = postprocess(
      values_out, elapsed_ms(preprocess_start, preprocess_end),
      elapsed_ms(inference_start, inference_end));
  prediction.postprocess_ms = elapsed_ms(postprocess_start, Clock::now());
  return prediction;
}

std::size_t OnnxClassifier::input_bytes() const {
  return static_cast<std::size_t>(3) * image_size_ * image_size_ * sizeof(float);
}

}  // namespace oncology
