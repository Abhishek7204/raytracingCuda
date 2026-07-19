#ifndef RAY_H
#define RAY_H

#include "vect.cuh"

class ray {
  point3 org;
  vect dir;

public:
  __device__ ray() {}
  __device__ ray(const point3 &origin, const vect &direction)
      : org(origin), dir(direction) {}

  __device__ const point3 &origin() const { return org; }
  __device__ const vect &direction() const { return dir; }
  __device__ point3 at(double t) { return org + dir * t; }
};

__device__ color rayColor(const ray &r);

#endif
