#ifndef LOADER_HPP
#define LOADER_HPP

#include <cstdint>
#include <string>
#include <vector>

class MnistLoader {
private:
  uint32_t row, col;
  std::vector<std::vector<double>> images;
  std::vector<uint8_t> labels;
  std::vector<std::vector<double>> oneHotLabels;
public:
  MnistLoader(std::string imgPath, std::string labelPath);
  MnistLoader(std::string imgPath);
  void loadImages(std::string path);
  void loadLabels(std::string path);
  void loadSingleImage(std::string path);
  const std::vector<std::vector<double>> &getImages();
  const std::vector<uint8_t> &getLabels();
  const std::vector<std::vector<double>> &getOneHotLabels();
};

#endif
