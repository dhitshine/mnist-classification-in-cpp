add_library(header
  "loader.cpp"
  "utility.cpp"
  "nn.cpp"
)

target_include_directories(header PUBLIC "${PROJECT_SOURCE_DIR}/include")
