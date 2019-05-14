#define __MAKEMORE_TRIANGLE_CC__ 1
#include <stdio.h>
#include "triangle.hh"

namespace makemore {

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
