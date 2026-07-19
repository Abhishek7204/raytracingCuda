#include "ray.cuh"

__device__ color rayColor(const ray &r) {
  vect unitDirection = unit_vect(r.direction());
  auto a = 0.5 * (unitDirection.y() + 1.0);
  return (1 - a) * color(1, 1, 1) + a * color(0.5, 0.7, 1.0);
}
