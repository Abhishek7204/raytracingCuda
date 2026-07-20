#ifndef SPHEREH
#define SPHEREH

#include "hitable.cuh"

class sphere : public hitable {
public:
  __device__ sphere() {}
  __device__ sphere(vect cen, float r) : center(cen), radius(r) {};
  __device__ virtual bool hit(const ray &r, float tmin, float tmax,
                              hitRecord &rec) const override;
  vect center;
  float radius;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max,
                            hitRecord &rec) const {
  vect oc = r.origin() - center;
  float a = dotProduct(r.direction(), r.direction());
  float b = dotProduct(oc, r.direction());
  float c = dotProduct(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.contactPoint = r.at(rec.t);
      rec.hitNormal = (rec.contactPoint - center) / radius;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.contactPoint = r.at(rec.t);
      rec.hitNormal = (rec.contactPoint - center) / radius;
      return true;
    }
  }
  return false;
}

#endif
