#ifndef MATERIALH
#define MATERIALH

#include "curand_kernel.h"
#include "hitable.cuh"
#include "ray.cuh"
#include "rt_utility.cuh"
#include "vect.cuh"

__device__ vect reflect(const vect &v, const vect &n) {
  return v - 2.0f * dotProduct(v, n) * n;
}

class material {
public:
  __device__ virtual bool scatter(const ray &r_in, const hitRecord &rec,
                                  vect &attenuation, ray &scattered,
                                  curandState *local_rand_state) const = 0;
  __device__ virtual ~material() = default;
};

class lambertian : public material {
public:
  __device__ lambertian(const vect &a) : albedo(a) {}
  __device__ virtual bool scatter(const ray &r_in, const hitRecord &rec,
                                  vect &attenuation, ray &scattered,
                                  curandState *local_rand_state) const {
    vect target = rec.contactPoint + rec.hitNormal +
                  random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.contactPoint, target - rec.contactPoint);
    attenuation = albedo;
    return true;
  }
  __device__ virtual ~lambertian() = default;

  vect albedo;
};

class metal : public material {
public:
  __device__ metal(const vect &a, float f) : albedo(a) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
  }
  __device__ virtual bool scatter(const ray &r_in, const hitRecord &rec,
                                  vect &attenuation, ray &scattered,
                                  curandState *local_rand_state) const {
    vect reflected = reflect(unitVector(r_in.direction()), rec.hitNormal);
    scattered = ray(rec.contactPoint,
                    reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dotProduct(scattered.direction(), rec.hitNormal) > 0.0f);
  }
  __device__ virtual ~metal() = default;
  vect albedo;
  float fuzz;
};

#endif
