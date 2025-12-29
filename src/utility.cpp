#include "utility.hpp"

Matrix::Matrix(size_t row_, size_t col_) : row(row_), col(col_), data(row * col, 0.0){}

Matrix::Matrix(const std::vector<std::vector<double>> &m) : row(m.size()), col(m.empty() ? 0 : m[0].size()), data(row * col){
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      data[i * col + j] = m[i][j];
    }
  }
}

const double &Matrix::operator()(size_t r, size_t c) const {    // read-only
  assert(r < row && c < col);
  return data[r * col + c];
}

double &Matrix::operator()(size_t r, size_t c){       // write
  assert(r < row && c < col);
  return data[r * col + c];
}

Matrix Matrix::operator-(const Matrix &other) const {     
  assert(row == other.row && col == other.col);
  Matrix res(row, col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      res(i, j) = (*this)(i, j) - other(i, j);
    }
  }
  return res;
}

Matrix &Matrix::operator-=(const Matrix &other) {     
  assert(row == other.row && col == other.col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      (*this)(i, j) -= other(i, j);
    }
  }
  return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {     
  assert(row == other.row && col == other.col);
  Matrix res(row, col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      res(i, j) = (*this)(i, j) + other(i, j);
    }
  }
  return res;
}

Matrix &Matrix::operator+=(const Matrix &other) {     
  assert(row == other.row && col == other.col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      (*this)(i, j) += other(i, j);
    }
  }
  return *this;
}

Matrix Matrix::operator*(const Matrix &other) const {
  Matrix res(row, other.col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < other.col; j++){
      double sum = 0.0;
      for(size_t k = 0; k < col; k++){
        sum += (*this)(i, k) * other(k, j);
      }
      res(i, j) = sum;
    }
  }
  return res;
}

Matrix Matrix::operator*(double scalar) const {
  Matrix res(row, col);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      res(i, j) = scalar * (*this)(i, j);
    }
  }
  return res;
}

Matrix operator*(double scalar, const Matrix &m) {
  return m * scalar;
}

Matrix Matrix::transpose() const {
  Matrix matTrans(col, row);
  for(size_t i = 0; i < row; i++){
    for(size_t j = 0; j < col; j++){
      matTrans(j, i) = (*this)(i, j);
    }
  }
  return matTrans;
}

Matrix Matrix::hadamard(const Matrix &other) const {
  assert(row == other.row && col == other.col);
  Matrix res(row, col);
  for(size_t i = 0; i < row; ++i){
    for(size_t j = 0; j < col; ++j){
      res(i, j) = (*this)(i, j) * other(i, j);
    }
  }
  return res;
}
