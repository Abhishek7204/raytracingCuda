#ifndef HITABLEH
#define HITABLEH

#include "interval.cuh"
#include "ray.cuh"

class hitRecord {
public:
  point3 contactPoint;
  vect hitNormal;
  double t;
};

class hitable {
public:
  __device__ virtual bool hit(const ray &r, interval ray_t,
                              hitRecord &rec) const = 0;
  __device__ virtual ~hitable() = default;
};

#endif
