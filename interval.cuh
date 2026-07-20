#ifndef INTERVAL_H
#define INTERVAL_H

#include <limits>

const double infinity = std::numeric_limits<double>::infinity();
class interval {

public:
  double iMin, iMax;
  __host__ __device__ interval() : iMin(+infinity), iMax(-infinity) {}

  __host__ __device__ interval(double iMin, double iMax)
      : iMin(iMin), iMax(iMax) {}

  __host__ __device__ double size() const { return iMax - iMin; }

  __host__ __device__ bool contains(double x) const {
    return iMin <= x && iMax >= x;
  }

  __host__ __device__ bool surrounds(double x) const {
    return iMin < x && x < iMax;
  }

  static const interval empty, universe;
};

const interval interval::empty = interval(infinity, -infinity);
const interval interval::universe = interval(-infinity, infinity);
#endif
