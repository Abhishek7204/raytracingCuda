#ifndef RT_UTILITY_H
#define RT_UTILITY_H

#include "vect.cuh"
#include <cmath>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

const double pi = 3.1415926535897932385;

#define RANDVECt                                                               \
  vect(curand_uniform(local_rand_state), curand_uniform(local_rand_state),     \
       curand_uniform(local_rand_state))

__device__ vect random_in_unit_sphere(curandState *local_rand_state) {
  vect p;
  do {
    p = 2.0f * RANDVECt - vect(1, 1, 1);
  } while (p.lenSquared() >= 1.0f);
  return p;
}

__host__ __device__ inline double degreeToRadian(double degrees) {
  return degrees * pi / 180.0;
}

#endif
