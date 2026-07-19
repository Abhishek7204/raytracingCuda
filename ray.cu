#include "ray.cuh"
#include "vect.cuh"

__device__ double hitSphere(const ray &r, const point3 &center, double radius) {
  // (C - P).(C - P) = (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2 = r^2
  // (C - (Q + dt)).(C - (Q + dt)) = r^2
  // t^2 d.d - 2td.(C-Q)+(C-Q).(C-Q) - r^2 = 0
  // a = d.d , b = -2d.(C-Q) , c = (C-Q).(C-Q) - r^2
  // discriminant = sqrt(b^2 - 4ac) no of roots is no of points touching sphere
  double a = dotProduct(r.direction(), r.direction());
  double b = -2 * dotProduct(r.direction(), center - r.origin());
  double c =
      dotProduct(center - r.origin(), center - r.origin()) - radius * radius;
  double discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return -1.0;
  return (-b - sqrt(discriminant)) / (2.0 * a);
}

__device__ color rayColor(const ray &r) {
  double t = hitSphere(r, point3(0, 0, -1), 0.5);
  if (t > 0) {
    vect hitNormal(unitVector(r.at(t) - vect(0, 0, -1)));
    return 0.5 * color(hitNormal.x() + 1, hitNormal.y() + 1, hitNormal.z() + 1);
  }
  vect unitDirection = unitVector(r.direction());
  auto a = 0.5 * (unitDirection.y() + 1.0);
  return (1 - a) * color(1, 1, 1) + a * color(0.5, 0.7, 1.0);
}
