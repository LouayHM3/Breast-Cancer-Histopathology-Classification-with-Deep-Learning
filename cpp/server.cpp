#include "inference.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

namespace {

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      escaped += '\\';
    }
    escaped += character;
  }
  return escaped;
}

std::string prediction_json(const oncology::Prediction& prediction) {
  std::ostringstream json;
  json << "{\"label\":\"" << json_escape(prediction.label)
       << "\",\"confidence\":" << prediction.confidence
       << ",\"benign_probability\":" << prediction.benign_probability
       << ",\"malignant_probability\":" << prediction.malignant_probability
       << ",\"timing_ms\":{\"preprocess\":" << prediction.preprocess_ms
       << ",\"inference\":" << prediction.inference_ms
       << ",\"postprocess\":" << prediction.postprocess_ms << "}}";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: oncology_server <classifier.onnx>\n";
    return EXIT_FAILURE;
  }
  try {
    auto classifier = std::make_shared<oncology::OnnxClassifier>(argv[1]);
    httplib::Server server;
    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
      response.set_content("{\"status\":\"ok\"}", "application/json");
    });
    server.Post("/v1/classify",
                [classifier](const httplib::Request& request,
                             httplib::Response& response) {
                  if (!request.has_file("image")) {
                    response.status = 400;
                    response.set_content("{\"error\":\"missing image field\"}",
                                         "application/json");
                    return;
                  }
                  try {
                    const auto file = request.get_file_value("image");
                    const std::vector<unsigned char> bytes(file.content.begin(),
                                                           file.content.end());
                    const cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
                    response.set_content(
                        prediction_json(classifier->classify(image)),
                        "application/json");
                  } catch (const std::exception& error) {
                    response.status = 422;
                    response.set_content(
                        std::string("{\"error\":\"") + json_escape(error.what()) +
                            "\"}",
                        "application/json");
                  }
                });
    std::cout << "Listening on http://127.0.0.1:8080\n";
    server.listen("127.0.0.1", 8080);
  } catch (const std::exception& error) {
    std::cerr << "Startup failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
