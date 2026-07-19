#include "vect.cuh"
#include <cmath>

__host__ __device__ vect vect::operator-() const {
  return vect(-v[0], -v[1], -v[2]);
};

__host__ __device__ vect vect::operator+=(const vect &vec) {
  v[0] += vec[0];
  v[1] += vec[1];
  v[2] += vec[2];
  return *this;
};

__host__ __device__ vect vect::operator*=(const vect &vec) {
  v[0] *= vec[0];
  v[1] *= vec[1];
  v[2] *= vec[2];
  return *this;
};

__host__ __device__ vect vect::operator-=(const vect &vec) {
  v[0] -= vec[0];
  v[1] -= vec[1];
  v[2] -= vec[2];
  return *this;
};

__host__ __device__ vect vect::operator/=(double s) {
  v[0] /= s;
  v[1] /= s;
  v[2] /= s;
  return *this;
};

__host__ __device__ double vect::len_squared() const {
  return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
};

__host__ __device__ double vect::len() const {
  return std::sqrt(len_squared());
};
