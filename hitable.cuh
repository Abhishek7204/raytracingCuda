#ifndef HITABLEH
#define HITABLEH

#include "ray.cuh"

class hitRecord {
public:
  point3 contactPoint;
  vect hitNormal;
  double t;
};

class hitable {
public:
  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hitRecord &rec) const = 0;
};

#endif
