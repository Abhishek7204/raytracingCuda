#ifndef RT_UTILITY_H
#define RT_UTILITY_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

const double pi = 3.1415926535897932385;

__host__ __device__ inline double degreeToRadian(double degrees) {
  return degrees * pi / 180.0;
}

#endif
