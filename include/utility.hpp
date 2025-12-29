#ifndef UTILITY_HPP
#define UTILITY_HPP
#include <cassert>
#include <cstddef>
#include <vector>

class Matrix { // flattened matrix to 1d vector
public:
  size_t row, col;
  std::vector<double> data;
  Matrix(size_t row_, size_t col_);
  Matrix(const std::vector<std::vector<double>> &m);
  const double &operator()(size_t r, size_t c) const;
  double &operator()(size_t r, size_t c);  
  Matrix operator-(const Matrix &other) const;
  Matrix &operator-=(const Matrix &other);
  Matrix operator+(const Matrix &other) const;
  Matrix &operator+=(const Matrix &other);
  Matrix operator*(const Matrix &other) const;
  Matrix operator*(double scalar) const;
  friend Matrix operator*(double scalar, const Matrix &m);
  Matrix transpose() const;
  Matrix hadamard(const Matrix &other) const;
};

#endif // !UTILITY_HPP
