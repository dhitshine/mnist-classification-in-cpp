#include "nn.hpp"

double kaimingNormalInit(size_t inFeatures){         // for ReLU
  static std::mt19937 rng(std::random_device{}());
  double stddev = std::sqrt(2.0 / static_cast<double>(inFeatures));
  std::normal_distribution<double> dist(0, stddev);
  return dist(rng);
}

double xavierNormalInit(size_t inFeatures, size_t outFeatures){  // for softmax
  static std::mt19937 rng(std::random_device{}());
  double stddev = std::sqrt(2.0 / static_cast<double>(inFeatures + outFeatures));
  std::normal_distribution<double> dist(0, stddev);
  return dist(rng);
}

Matrix relu(const Matrix &z){
  Matrix res(z.row, z.col);
  for(size_t i = 0; i < z.row; i++){
    for(size_t j = 0; j < z.col; j++){
      double val = z(i, j);
      if(val <= 0){
        res(i, j) = 0;
      }else{
        res(i, j) = val;
      }
    }
  }
  return res;
}

Matrix reluDerivative(const Matrix &z){
  Matrix res(z.row, z.col);
  for(size_t i = 0; i < z.row; i++){
    for(size_t j = 0; j < z.col; j++){
      double val = z(i, j);
      if(val <= 0){
        res(i, j) = 0;
      }else{
        res(i, j) = 1;
      }
    }
  }
  return res;
}

Matrix softmax(const Matrix &z){
  Matrix res(z.row, z.col);
  Matrix out(z.row, z.col);
  double max_z = *std::max_element(z.data.begin(), z.data.end());
  for(size_t i = 0; i < z.row; i++){
    for(size_t j = 0; j < z.col; j++){
      out(i, j) = std::exp(z(i, j) - max_z);
    }
  }
  double sum = std::accumulate(out.data.begin(), out.data.end(), 0.0);
  sum = std::max(sum, epsilon);
  for(size_t i = 0; i < z.row; i++){
    for(size_t j = 0; j < z.col; j++){
      res(i, j) = (out(i, j) / sum);
    }
  }
  return res;
}

double crossEntropy(const Matrix &p, const Matrix &y){    // p = predictions, y = true label
  // L(p;y) = - \sum y log p 
  double loss = 0;
  for(size_t i = 0; i < p.row; i++){
    for(size_t j = 0; j < p.col; j++){
      double prob = std::max(p(i, j), epsilon);
      loss -= y(i, j) * std::log(prob);
    }
  }
  return loss;
}

Matrix getSample(const Matrix &sources, size_t idx){
  Matrix x(sources.col, 1);
  for(size_t i = 0; i < sources.col; i++){
    x(i, 0) = sources(idx, i);
  }
  return x;
}

Layer::Layer(size_t in, size_t out, activationFunction act = activationFunction::ReLU) : inFeatures(in), outFeatures(out), W(out, in), b(out, 1), z(out, 1), a(out, 1), vW(out, in), vb(out, 1){   // default is kaiming
    initLayer(act);
  }
  
void Layer::initLayer(activationFunction act){
  // weight
  if(act == activationFunction::ReLU){
    for(size_t i = 0; i < outFeatures; i++){
      for(size_t j = 0; j < inFeatures; j++){
        W(i, j) = kaimingNormalInit(inFeatures);
      }
    }
  }else if(act == activationFunction::Softmax){
    for(size_t i = 0; i < outFeatures; i++){
      for(size_t j = 0; j < inFeatures; j++){
        W(i, j) = xavierNormalInit(inFeatures, outFeatures);
      }
    }
  }
}

NeuralNetwork::NeuralNetwork(size_t input, size_t hidden, size_t output) : inputSize(input), hiddenSize(hidden), outputSize(output), hidden(inputSize, hiddenSize, activationFunction::ReLU), output(hiddenSize, outputSize, activationFunction::Softmax), x(input, 1){}

Matrix NeuralNetwork::forwardPropagate(const Matrix &input){
  x = input;
  hidden.z = hidden.W * x;
  hidden.z += hidden.b;
  hidden.a = relu(hidden.z);
  output.z = output.W * hidden.a;
  output.z += output.b;
  output.a = softmax(output.z);
  return output.a;
}
std::tuple<Matrix, Matrix, Matrix, Matrix> NeuralNetwork::backPropagate(const Matrix &y_true, const Matrix &y_pred){
  Matrix deltaOutput = y_pred - y_true;
  Matrix weightOutputError = deltaOutput * hidden.a.transpose();
  Matrix biasOutputError = deltaOutput;
  Matrix deltaHidden = (output.W.transpose() * deltaOutput).hadamard(reluDerivative(hidden.z));
  Matrix weightHiddenError = deltaHidden * x.transpose();
  Matrix biasHiddenError = deltaHidden;
  return {weightHiddenError, biasHiddenError, weightOutputError, biasOutputError};
}
void NeuralNetwork::training(const Matrix &images, const Matrix &trueLabels, size_t batchSize, size_t epochs, double learningRate){
  std::cout << "Starting Training..." << std::endl;

  static std::mt19937 rng(std::random_device{}());
  size_t imagesSize = images.row;
  std::vector<size_t> indices(imagesSize);
  std::iota(indices.begin(), indices.end(), 0);
  for(size_t i = 0; i < epochs; i++){
    std::shuffle(indices.begin(), indices.end(), rng);
    double totalLoss = 0.0;
    size_t correct = 0;
    for(uint32_t j = 0; j < imagesSize; j += batchSize){
      size_t currentBatchSize = std::min(batchSize, imagesSize - j);
      Matrix dw1Batch(hiddenSize, inputSize), db1Batch(hiddenSize, 1), dw2Batch(outputSize, hiddenSize), db2Batch(outputSize, 1); 
      for(size_t k = 0; k < currentBatchSize; k++){
        size_t idx = indices[j + k];
        Matrix x = getSample(images, idx);
        Matrix y = getSample(trueLabels, idx);
        Matrix output = forwardPropagate(x);
        auto [dw1, db1, dw2, db2] = backPropagate(y, output);

        dw1Batch += dw1;
        db1Batch += db1;
        dw2Batch += dw2;
        db2Batch += db2;
        
        size_t predClass = 0;
        size_t trueClass = 0;
        double maxPred = output(0, 0);
        double maxTrue = y(0, 0);
        for(size_t c = 1; c < outputSize; c++){   // argmax
          if(output(c, 0) > maxPred){
            maxPred = output(c, 0);
            predClass = c;
          }
          if(y(c, 0) > maxTrue) {
            maxTrue = y(c, 0);
            trueClass = c;
          }
        }
        if(predClass == trueClass){
          correct++;
        }
        totalLoss += crossEntropy(output, y);
      }
      double invBatch = 1.0 / static_cast<double>(currentBatchSize);
      dw1Batch = invBatch * dw1Batch;
      db1Batch = invBatch * db1Batch;
      dw2Batch = invBatch * dw2Batch;
      db2Batch = invBatch * db2Batch;

      // momentum
      double gamma = 0.9;
      hidden.vW = (gamma * hidden.vW) + (learningRate * dw1Batch);
      hidden.vb = (gamma * hidden.vb) + (learningRate * db1Batch);
      output.vW = (gamma * output.vW) + (learningRate * dw2Batch);
      output.vb = (gamma * output.vb) + (learningRate * db2Batch);
      hidden.W -= hidden.vW;
      hidden.b -= hidden.vb;
      output.W -= output.vW;
      output.b -= output.vb;
    }
    double avgLoss = totalLoss / static_cast<double>(imagesSize);
    double accuracy = static_cast<double>(correct) / static_cast<double>(imagesSize);
    std::cout << "Epoch " << i + 1 << "/" << epochs << " - Loss: " << std::fixed << std::setprecision(4) << avgLoss << ", Accuracy: " << std::fixed << std::setprecision(4) << accuracy << " (" << correct << "/" << imagesSize << ")" << std::endl;
  }
}

void NeuralNetwork::predict(const Matrix &testImages){
  Matrix x = getSample(testImages, 0);
  Matrix output = forwardPropagate(x);
  std::cout << "Probability predictions: " << std::endl;
  for(size_t i = 0; i < output.row; i++){
    for(size_t j = 0; j < output.col; j++){
      std::cout << i << ": " << output(i, j) << " ";
    }
    std::cout << std::endl;
  }
  size_t predClass = 0;
  double maxPred = output(0, 0);
  for(size_t c = 1; c < outputSize; c++){
    if(output(c, 0) > maxPred){
      maxPred = output(c, 0);
      predClass = c;
    }
  }
  std::cout << "Prediction class: " << predClass << std::endl;
}

void NeuralNetwork::evaluate(const Matrix &testImages, const Matrix &testLabels) {
  size_t testSize = testImages.row;
  size_t correct = 0;
  double totalLoss = 0.0;
  for(size_t i = 0; i < testSize; i++){
    Matrix x = getSample(testImages, i);
    Matrix y = getSample(testLabels, i);
    Matrix output = forwardPropagate(x);
    totalLoss += crossEntropy(output, y);
    size_t predClass = 0;
    size_t trueClass = 0;
    double maxPred = output(0, 0);
    double maxTrue = y(0, 0);
    for(size_t c = 1; c < outputSize; c++){
      if(output(c, 0) > maxPred){
        maxPred = output(c, 0);
        predClass = c;
      }
      if(y(c, 0) > maxTrue){
        maxTrue = y(c, 0);
        trueClass = c;
      }
    }
    /*
    std::cout << "Prediksi: ";
    for(size_t c = 0; c < outputSize; c++){
      std::cout << std::setprecision(2) << std::fixed <<  output(c, 0) << " ";
    }
    std::cout << "Label Asli: ";
    for(size_t c = 0; c < outputSize; c++){
      std::cout << std::setprecision(4) << std::fixed << y(c, 0) << " ";
    }
    std::cout << std::endl;
    */
    if(predClass == trueClass){
      correct++;
    }
  }
  double accuracy = static_cast<double>(correct) / static_cast<double>(testSize);
  double avgLoss = totalLoss / static_cast<double>(testSize);
  std::cout << "\nTest Result:\n" << std::endl;
  std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << avgLoss << ", Test Accuracy: " << accuracy << " (" << correct << "/" << testSize << ")" << std::endl;
}
