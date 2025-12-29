#ifndef NN_HPP
#define NN_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <random>
#include <vector>
#include <cstddef>
#include <ostream>
#include "utility.hpp"
const double epsilon = 1e-9;

double kaimingNormalInit(size_t inFeatures);
double xavierNormalInit(size_t inFeatures, size_t outFeatures);
Matrix matrixDot(const Matrix &a, const Matrix &b);
Matrix relu(const Matrix &z);
Matrix reluDerivative(const Matrix &z);
Matrix softmax(const Matrix &z);
double crossEntropy(const Matrix &p, const Matrix &y);
Matrix getSample(const Matrix &sources, size_t idx);

enum class activationFunction {
  ReLU,
  Softmax
};

class Layer {
private:
  size_t inFeatures, outFeatures;
public:
  Matrix W;   // weight
  Matrix b;   // bias
  Matrix z;   // pre activation
  Matrix a;   // post activation
  Matrix vW;  // weight momentum
  Matrix vb;  // bias momentum
  Layer(size_t in, size_t out, activationFunction act);
private:
  void initLayer(activationFunction act);
};


class NeuralNetwork {
private:
  size_t inputSize, hiddenSize, outputSize;
  Layer hidden;
  Layer output;
  Matrix x;       // input
public:
  NeuralNetwork(size_t input, size_t hidden, size_t output);
  Matrix forwardPropagate(const Matrix &input);
  std::tuple<Matrix, Matrix, Matrix, Matrix> backPropagate(const Matrix &y_true, const Matrix &y_pred);
  void training(const Matrix &images, const Matrix &trueLabels, size_t batchSize, size_t epochs, double learningRate);
  void evaluate(const Matrix &testImages, const Matrix &testLabels);
  void predict(const Matrix &testImages);
};

#endif
