#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include "loader.hpp"

MnistLoader::MnistLoader(std::string imgPath, std::string labelPath){
  loadImages(imgPath);
  loadLabels(labelPath);
}

MnistLoader::MnistLoader(std::string imgPath){
  loadImages(imgPath);
}

void MnistLoader::loadImages(std::string path){
  uint32_t magic, imageNum;
  std::ifstream file(path, std::ios::binary);
  if(!file.is_open()){
    throw std::runtime_error("Cannot opening file: " + path);
  }
  file.read(reinterpret_cast<char*> (&magic), sizeof(magic));    // read hanya nerima char*
  file.read(reinterpret_cast<char*> (&imageNum), sizeof(imageNum));
  file.read(reinterpret_cast<char*> (&row), sizeof(row));
  file.read(reinterpret_cast<char*> (&col), sizeof(col));
  magic = __builtin_bswap32(magic);                             // swap from big endian to little endian
  imageNum = __builtin_bswap32(imageNum);
  // std::cout << imageNum << std::endl;
  row = __builtin_bswap32(row);
  // std::cout << row << std::endl;
  col = __builtin_bswap32(col);
  // std::cout << col << std::endl;

  images.resize(imageNum);
  uint32_t imageSize = row * col;
  std::vector<uint8_t> buffer(imageSize);
  for(uint32_t i = 0; i < imageNum; i++){
    file.read(reinterpret_cast<char*>(buffer.data()), imageSize);   // buffer.data() -> alamat buffer atau &buffer[0]
    images[i].resize(imageSize);
    for(uint32_t j = 0; j < imageSize; j++){
      images[i][j] = (buffer[j] / 255.0);     // normalisasi per pixel
    }
  }
}

void MnistLoader::loadLabels(std::string path){
  uint32_t magic, labelNum;
  std::ifstream file(path, std::ios::binary);
  if(!file.is_open()){
    throw std::runtime_error("Cannot opening file: " + path);
  }
  file.read(reinterpret_cast<char*> (&magic), sizeof(magic));
  file.read(reinterpret_cast<char*> (&labelNum), sizeof(labelNum));
  magic = __builtin_bswap32(magic);
  labelNum = __builtin_bswap32(labelNum);
  labels.resize(labelNum);
  file.read(reinterpret_cast<char*> (labels.data()), labelNum);
  oneHotLabels.resize(labelNum, std::vector<double> (10, 0.0));
  for(uint32_t i = 0; i < labelNum; i++){
    uint8_t label = labels[i];
    oneHotLabels[i][label] = 1.0;
  }
}

const std::vector<std::vector<double>> &MnistLoader::getImages(){
  return images;
}

const std::vector<uint8_t> &MnistLoader::getLabels(){
  return labels;
}

const std::vector<std::vector<double>> &MnistLoader::getOneHotLabels(){
  return oneHotLabels;
}
/*
int main(){
  MnistLoader mn("../mnist/train/train-images.idx3-ubyte");
  const std::vector<std::vector<double>> a = mn.getImages();
  for(size_t i = 0; i < 784; i++){
    for(size_t j = 0; j < 1; j++){
      std::cout << a[i][j] << " ";
    }
    std::cout << std::endl;
  }
}
*/
/*
File idx3-ubyte menggunakan byte ordering Big-Endian, yaitu membaca dari MSB ke LSB
*/
