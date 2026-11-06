# C++ ONNX deployment

This directory contains a native inference service for the **classification**
ONNX export. It uses OpenCV for image decoding/preprocessing, ONNX Runtime for
inference, and `cpp-httplib` for the HTTP API.

## Build dependencies

- CMake 3.20+
- C++17 compiler
- OpenCV (`core`, `imgcodecs`, `imgproc`)
- ONNX Runtime with its CMake package configuration
- Internet access on the first CMake configure, to fetch `cpp-httplib`

Configure ONNX Runtime explicitly if it is not installed system-wide:

```powershell
cmake -S cpp -B build `
  -Donnxruntime_DIR="C:\path\to\onnxruntime\lib\cmake\onnxruntime"
cmake --build build --config Release
```

## Run the service

```powershell
.\build\Release\oncology_server.exe artifacts\resnet18.onnx
curl.exe http://127.0.0.1:8080/health
curl.exe -F "image=@sample.png" http://127.0.0.1:8080/v1/classify
```

The API returns the class, both class probabilities, and preprocessing,
inference, and postprocessing timings. Invalid or missing images return an
explicit `4xx` response.

## Benchmark

```powershell
.\build\Release\oncology_benchmark.exe artifacts\resnet18.onnx sample.png
```

The benchmark performs warm-up iterations and reports mean, minimum, and
maximum end-to-end latency plus input tensor memory. Run it on the target
hardware and record results rather than assuming notebook performance.

## Segmentation

The server currently supports classification because the available BreaKHis
download has no segmentation masks and therefore no trained segmentation ONNX
model. A future segmentation endpoint should use a separate mask-trained model
and return a validated mask/overlay contract instead of treating classification
heatmaps as segmentation.
