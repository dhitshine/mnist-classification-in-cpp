cmake_minimum_required(VERSION 3.15)

project(
  mnist-classification
  VERSION 1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # deteksi path include dll

add_compile_options(-Wall -Wextra -Weffc++ -Wpedantic -O2)

add_subdirectory(src)

add_subdirectory(apps)
