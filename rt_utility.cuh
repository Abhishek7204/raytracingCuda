#ifndef RT_UTILITY_H
#define RT_UTILITY_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degreeToRadian(double degrees) { return degrees * pi / 180.0; }

#endif
