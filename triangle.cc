#define __MAKEMORE_TRIANGLE_CC__ 1
#include <stdio.h>
#include "triangle.hh"

namespace makemore {

bool Triangle::contains(const Point &a) {
  double d;

  d = (a.x - p.x) * (q.y - p.y) - (a.y - p.y) * (q.x - p.x);
  if (d > 0) return true;
  d = (a.x - q.x) * (r.y - q.y) - (a.y - q.y) * (r.x - q.x);
  if (d > 0) return true;
  d = (a.x - r.x) * (p.y - r.y) - (a.y - r.y) * (p.x - r.x);
  if (d > 0) return true;

  return false;
}

}

#if MAIN

using namespace makemore;

int main() {
  Triangle t1(Point(0, 0), Point(1, 0), Point(0, 1));
  Triangle t2(Point(0, 0), Point(2, 0), Point(0, 1));

  Point p(0.5, 0.5);
  Point q = trimap(p, t1, t2);
  printf("(%lf,%lf)\n", p.x, p.y);
  printf("(%lf,%lf)\n", q.x, q.y);
}

#endif
