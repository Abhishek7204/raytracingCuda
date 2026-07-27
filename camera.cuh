#ifndef CAMERAH
#define CAMERAH

#include "ray.cuh"

class camera {
public:
  __device__ camera(vect lookfrom, vect lookat, vect vup, float vfov,
                    float aspect) {
    // vfov is top to bottom in degrees
    vect u, v, w;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unitVector(lookfrom - lookat);
    u = unitVector(crossProduct(vup, w));
    v = crossProduct(w, u);
    lower_left_corner = origin - half_width * u - half_height * v - w;
    horizontal = 2 * half_width * u;
    vertical = 2 * half_height * v;
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
