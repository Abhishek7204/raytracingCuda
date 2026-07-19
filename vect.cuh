#ifndef VECT_H
#define VECT_H

#include <iostream>

using namespace std;

class vect {
  double v[3];

public:
  __host__ __device__ vect() : v{0, 0, 0} {}
  __host__ __device__ vect(double v1, double v2, double v3) : v{v1, v2, v3} {}

  __host__ __device__ double x() const { return v[0]; }
  __host__ __device__ double y() const { return v[1]; }
  __host__ __device__ double z() const { return v[2]; }
  __host__ __device__ double &operator[](int i) { return v[i]; }
  __host__ __device__ double operator[](int i) const { return v[i]; }

  __host__ __device__ vect operator-() const;
  __host__ __device__ vect operator+=(const vect &vec);
  __host__ __device__ vect operator*=(const vect &vec);
  __host__ __device__ vect operator-=(const vect &vec);
  __host__ __device__ vect operator/=(double s);

  __host__ __device__ double len_squared() const;
  __host__ __device__ double len() const;
};

using point3 = vect;
using color = vect;

inline ostream &operator<<(ostream &out, const vect &vec) {
  return out << vec[0] << ' ' << vec[1] << ' ' << vec[2];
}

__host__ __device__ inline vect operator+(const vect &vec1, const vect &vec2) {
  return vect(vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]);
}

__host__ __device__ inline vect operator-(const vect &vec1, const vect &vec2) {
  return vect(vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]);
}

__host__ __device__ inline vect operator*(const vect &vec1, const vect &vec2) {
  return vect(vec1[0] * vec2[0], vec1[1] * vec2[1], vec1[2] * vec2[2]);
}

__host__ __device__ inline vect operator*(const vect &vec, double s) {
  return vect(vec[0] * s, vec[1] * s, vec[2] * s);
}

__host__ __device__ inline vect operator*(double s, const vect &vec) {
  return vect(vec[0] * s, vec[1] * s, vec[2] * s);
}

__host__ __device__ inline vect operator/(const vect &vec, double s) {
  return vect(vec[0] / s, vec[1] / s, vec[2] / s);
}

__host__ __device__ inline vect operator/(double s, const vect &vec) {
  return vect(vec[0] / s, vec[1] / s, vec[2] / s);
}

__host__ __device__ inline double dot(const vect &vec1, const vect &vec2) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

__host__ __device__ inline vect cross(const vect &vec1, const vect &vec2) {
  return vect(vec1[1] * vec2[2] - vec1[2] * vec2[1],
              vec1[2] * vec2[0] - vec1[0] * vec2[2],
              vec1[0] * vec2[1] - vec1[1] * vec2[0]);
}

__host__ __device__ inline vect unit_vect(const vect &vec) {
  return vec / vec.len();
}

#endif
