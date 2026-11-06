#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

namespace oncology {

struct Prediction {
  std::string label;
  float confidence;
  float benign_probability;
  float malignant_probability;
  double preprocess_ms;
  double inference_ms;
  double postprocess_ms;
};

class OnnxClassifier {
 public:
  OnnxClassifier(const std::string& model_path, int image_size = 224);

  Prediction classify(const cv::Mat& image) const;
  std::size_t input_bytes() const;

 private:
  std::vector<float> preprocess(const cv::Mat& image) const;
  Prediction postprocess(const std::vector<float>& logits,
                         double preprocess_ms,
                         double inference_ms) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_name_;
  std::string output_name_;
  int image_size_;
};

}  // namespace oncology
