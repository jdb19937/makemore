#ifndef __MAKEMORE_TRIANGLE_HH__
#define __MAKEMORE_TRIANGLE_HH__ 1

#include "point.hh"

namespace makemore {

struct Triangle {
  Triangle() { }
  Triangle(double px, double py, double qx, double qy, double rx, double ry) : p(px, py), q(qx, qy), r(rx, ry) { }
  Triangle(const Point &_p, const Point &_q, const Point &_r) : p(_p), q(_q), r(_r) { }
  Point p, q, r;
};

inline Point trimap(const Point &a, const Triangle &from, const Triangle &to) {
  double denom = (-(from.q.y*from.r.x) + from.p.y*(-from.q.x + from.r.x) + from.p.x*(from.q.y - from.r.y) + from.q.x*from.r.y);
  assert(denom != 0.0);

  return Point(
    (-(to.p.x*from.q.y*from.r.x) + from.p.y*to.q.x*from.r.x + to.p.x*from.q.x*from.r.y - from.p.x*to.q.x*from.r.y - from.p.y*from.q.x*to.r.x + 
    from.p.x*from.q.y*to.r.x + a.y*(from.p.x*to.q.x - to.q.x*from.r.x + to.p.x*(-from.q.x + from.r.x) - from.p.x*to.r.x + from.q.x*to.r.x) + 
    a.x*(to.p.x*from.q.y - from.p.y*to.q.x - to.p.x*from.r.y + to.q.x*from.r.y + from.p.y*to.r.x - from.q.y*to.r.x))/denom
    ,
    (-(to.p.y*from.q.y*from.r.x) + from.p.y*to.q.y*from.r.x + to.p.y*from.q.x*from.r.y - from.p.x*to.q.y*from.r.y - from.p.y*from.q.x*to.r.y +
    from.p.x*from.q.y*to.r.y + a.y*(from.p.x*to.q.y - to.q.y*from.r.x + to.p.y*(-from.q.x + from.r.x) - from.p.x*to.r.y + from.q.x*to.r.y) + 
    a.x*(to.p.y*from.q.y - from.p.y*to.q.y - to.p.y*from.r.y + to.q.y*from.r.y + from.p.y*to.r.y - from.q.y*to.r.y))/denom
  );
}

}

#endif
