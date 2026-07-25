#ifndef SPHEREH
#define SPHEREH

#include "hitable.cuh"
#include "material.cuh"

class sphere : public hitable {
public:
  __device__ sphere() {}
  __device__ sphere(vect cen, float r, material *m)
      : center(cen), radius(r), mat(m) {};
  __device__ virtual bool hit(const ray &r, interval ray_t,
                              hitRecord &rec) const override;
  vect center;
  double radius;
  material *mat;
};

__device__ bool sphere::hit(const ray &r, interval ray_t,
                            hitRecord &rec) const {
  vect oc = r.origin() - center;
  float a = dotProduct(r.direction(), r.direction());
  float b = dotProduct(oc, r.direction());
  float c = dotProduct(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < ray_t.iMax && temp > ray_t.iMin) {
      rec.t = temp;
      rec.contactPoint = r.at(rec.t);
      rec.hitNormal = (rec.contactPoint - center) / radius;
      rec.mat = mat;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < ray_t.iMax && temp > ray_t.iMin) {
      rec.t = temp;
      rec.contactPoint = r.at(rec.t);
      rec.hitNormal = (rec.contactPoint - center) / radius;
      rec.mat = mat;
      return true;
    }
  }
  return false;
}

#endif
