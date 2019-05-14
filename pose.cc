#define __MAKEMORE_POSE_CC__ 1

#include <stdlib.h>
#include <string.h>

#include <math.h>

#include <string>

#include "pose.hh"

namespace makemore {

const Pose Pose::STANDARD(128, 160, 64, 0.0, 1.0, 0.0);

static inline double pdl(const Point &p0, const Point &p1, const Point &p2) {
  double d2 = (p2.y - p1.y) * (p2.y - p1.y) + (p2.x - p1.x) * (p2.x - p1.x);
  assert(d2 > 0);
  return ((p2.y - p1.y) * p0.x - (p2.x - p1.x) * p0.y + p2.x * p1.y - p2.y * p1.x) / sqrt(d2);
}

Pose::Pose(const Triangle &tri) {
  Point mid = tri.p * 0.5 + tri.q * 0.5;
  center = mid * 0.5 + tri.r * 0.5;

  Point pq = tri.q - tri.p;
  scale = pq.abs();
  assert(scale > 0);

  angle = ::atan2(-pq.y, pq.x);

  stretch = pdl(tri.r, tri.q, tri.p) / scale;

  Point ns = mid + Point(-pq.y, pq.x);
  skew = pdl(tri.r, mid, ns) / scale;
}

Triangle Pose::mark() const {
  Point pq = Point(scale * cos(angle), scale * -sin(angle));
  Point r = center + Point(-pq.y, pq.x) * 0.5 * stretch + pq * 0.5 * skew;
  Point mid = center * 2.0 - r;
  Point p = mid - pq * 0.5;
  Point q = mid + pq * 0.5;
  return Triangle(p, q, r);
}

}
