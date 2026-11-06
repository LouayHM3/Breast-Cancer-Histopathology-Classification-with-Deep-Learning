cmake_minimum_required(VERSION 3.20)
project(oncology_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(onnxruntime CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.16.3
)
FetchContent_MakeAvailable(httplib)

add_library(oncology_inference inference.cpp)
target_include_directories(oncology_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(oncology_inference PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

if(MSVC)
  target_compile_options(oncology_inference PRIVATE /W4 /permissive-)
else()
  target_compile_options(oncology_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(oncology_server server.cpp)
target_link_libraries(oncology_server PRIVATE oncology_inference httplib::httplib)

add_executable(oncology_benchmark benchmark.cpp)
target_link_libraries(oncology_benchmark PRIVATE oncology_inference)
