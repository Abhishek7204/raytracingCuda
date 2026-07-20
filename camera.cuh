#ifndef CAMERAH
#define CAMERAH

#include "ray.cuh"

class camera {
public:
  __device__ camera() {
    lower_left_corner = vect(-2.0, -1.0, -1.0);
    horizontal = vect(4.0, 0.0, 0.0);
    vertical = vect(0.0, 2.0, 0.0);
    origin = vect(0.0, 0.0, 0.0);
  }
  __device__ ray get_ray(float u, float v) {
    return ray(origin,
               lower_left_corner + u * horizontal + v * vertical - origin);
  }

  vect origin;
  vect lower_left_corner;
  vect horizontal;
  vect vertical;
};

#endif
