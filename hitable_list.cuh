#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.cuh"

class hitable_list : public hitable {
public:
  __device__ hitable_list() {}
  __device__ hitable_list(hitable **l, int n) {
    list = l;
    list_size = n;
  }
  __device__ virtual bool hit(const ray &r, interval ray_t,
                              hitRecord &rec) const override;

  __device__ virtual ~hitable_list() = default;
  hitable **list;
  int list_size;
};

__device__ bool hitable_list::hit(const ray &r, interval ray_t,
                                  hitRecord &rec) const {
  hitRecord temp_rec;
  bool hit_anything = false;
  double closest_so_far = ray_t.iMax;
  for (int i = 0; i < list_size; i++) {
    if (list[i]->hit(r, interval(ray_t.iMin, closest_so_far), temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

#endif
