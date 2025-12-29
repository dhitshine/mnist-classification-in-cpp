#include "nn.hpp"
#include "loader.hpp"

int main(){
  MnistLoader trainLoad("mnist/train/train-images.idx3-ubyte", "mnist/train/train-labels.idx1-ubyte");
  Matrix trainImages = trainLoad.getImages();
  Matrix trainLabels = trainLoad.getOneHotLabels();
  MnistLoader testLoad("mnist/test/t10k-images.idx3-ubyte", "mnist/test/t10k-labels.idx1-ubyte");
  Matrix testImages = testLoad.getImages();
  Matrix testLabels = testLoad.getOneHotLabels();

  NeuralNetwork nn(784, 128, 10);
  nn.training(trainImages, trainLabels, 64, 10, 0.01);    // batch, epochs, learning rate
  nn.evaluate(testImages, testLabels);
  MnistLoader p1("../mnist/misc/img2g.idx3-ubyte");
  MnistLoader p2("../mnist/misc/img3g.idx3-ubyte");
  MnistLoader p3("../mnist/misc/img5g.idx3-ubyte");
  MnistLoader p4("../mnist/misc/img6g.idx3-ubyte");
  MnistLoader p5("../mnist/misc/img7g.idx3-ubyte");
  MnistLoader p6("../mnist/misc/img8g.idx3-ubyte");
  MnistLoader p7("../mnist/misc/img9g.idx3-ubyte");
  Matrix img2 = p1.getImages();   // gambar 2
  nn.predict(img2);
  Matrix img3 = p2.getImages();  // gambar 3
  nn.predict(img3);
  Matrix img5 = p3.getImages(); // gambar 5
  nn.predict(img5);
  Matrix img6 = p4.getImages(); // gambar 6
  nn.predict(img6);
  Matrix img7 = p5.getImages(); // gambar 7
  nn.predict(img7);
  Matrix img8 = p6.getImages(); // gambar 8
  nn.predict(img8);
  Matrix img9 = p7.getImages(); // gambar 9
  nn.predict(img9);
}
