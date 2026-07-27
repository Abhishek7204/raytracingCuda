#ifndef MATERIALH
#define MATERIALH

#include "curand_kernel.h"
#include "hitable.cuh"
#include "ray.cuh"

__device__ float schlick(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vect &v, const vect &n, float ni_over_nt,
                        vect &refracted) {
  vect uv = unitVector(v);
  float dt = dotProduct(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else
    return false;
}

#define RANDvect                                                               \
  vect(curand_uniform(local_rand_state), curand_uniform(local_rand_state),     \
       curand_uniform(local_rand_state))

__device__ vect random_in_unit_sphere(curandState *local_rand_state) {
  vect p;
  do {
    p = 2.0f * RANDvect - vect(1, 1, 1);
  } while (p.lenSquared() >= 1.0f);
  return p;
}

__device__ vect reflect(const vect &v, const vect &n) {
  return v - 2.0f * dotProduct(v, n) * n;
}

class material {
public:
  __device__ virtual bool scatter(const ray &r_in, const hitRecord &rec,
                                  vect &attenuation, ray &scattered,
                                  curandState *local_rand_state) const = 0;
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
  vect albedo;
  float fuzz;
};

class dielectric : public material {
public:
  __device__ dielectric(float ri) : ref_idx(ri) {}
  __device__ virtual bool scatter(const ray &r_in, const hitRecord &rec,
                                  vect &attenuation, ray &scattered,
                                  curandState *local_rand_state) const {
    vect outward_normal;
    vect reflected = reflect(r_in.direction(), rec.hitNormal);
    float ni_over_nt;
    attenuation = vect(1.0, 1.0, 1.0);
    vect refracted;
    float reflect_prob;
    float cosine;
    if (dotProduct(r_in.direction(), rec.hitNormal) > 0.0f) {
      outward_normal = -rec.hitNormal;
      ni_over_nt = ref_idx;
      cosine =
          dotProduct(r_in.direction(), rec.hitNormal) / r_in.direction().len();
      cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    } else {
      outward_normal = rec.hitNormal;
      ni_over_nt = 1.0f / ref_idx;
      cosine =
          -dotProduct(r_in.direction(), rec.hitNormal) / r_in.direction().len();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
      reflect_prob = schlick(cosine, ref_idx);
    else
      reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = ray(rec.contactPoint, reflected);
    else
      scattered = ray(rec.contactPoint, refracted);
    return true;
  }

  float ref_idx;
};
#endif
