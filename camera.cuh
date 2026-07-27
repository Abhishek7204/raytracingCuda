#ifndef CAMERAH
#define CAMERAH

#include "curand_kernel.h"
#include "ray.cuh"

__device__ vect random_in_unit_disk(curandState *local_rand_state) {
  vect p;
  do {
    p = 2.0f * vect(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state), 0) -
        vect(1, 1, 0);
  } while (dotProduct(p, p) >= 1.0f);
  return p;
}

class camera {
public:
  __device__ camera(vect lookfrom, vect lookat, vect vup, float vfov,
                    float aspect, float aperture,
                    float focus_dist) { // vfov is top to bottom in degrees
    radius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unitVector(lookfrom - lookat);
    u = unitVector(crossProduct(vup, w));
    v = crossProduct(w, u);
    lower_left_corner = origin - half_width * focus_dist * u -
                        half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
  }
  __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
    vect rd = radius * random_in_unit_disk(local_rand_state);
    vect offset = u * rd.x() + v * rd.y();
    return ray(origin + offset, lower_left_corner + s * horizontal +
                                    t * vertical - origin - offset);
  }

  vect origin;
  vect lower_left_corner;
  vect horizontal;
  vect vertical;
  vect u, v, w;
  double radius;
};

#endif
