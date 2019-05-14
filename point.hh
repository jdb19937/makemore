#ifndef __MAKEMORE_POINT_HH__
#define __MAKEMORE_POINT_HH__ 1

#include <assert.h>
#include <math.h>

namespace makemore {

struct Point {
  double x, y;

  Point() : x(0.0), y(0.0) {
  }
  Point(double _x, double _y) : x(_x), y(_y) {
  }
  Point(const Point &p) {
    x = p.x;
    y = p.y;
  }

  Point &operator =(const Point &p) {
    x = p.x;
    y = p.y;
    return *this;
  }

  Point &operator +=(const Point &p) {
    x += p.x;
    y += p.y;
    return *this;
  }

  Point operator +(const Point &p) const {
    return Point(x + p.x, y + p.y);
  }

  Point &operator -=(const Point &p) {
    x -= p.x;
    y -= p.y;
    return *this;
  }

  Point operator -(const Point &p) const {
    return Point(x - p.x, y - p.y);
  }

  Point &operator *=(double a) {
    x *= a;
    y *= a;
    return *this;
  }

  Point operator *(double a) const {
    return Point(x * a, y * a);
  }

  Point &operator /=(double a) {
    assert(a != 0.0);
    x /= a;
    y /= a;
    return *this;
  }

  Point operator /(double a) const {
    assert(a != 0.0);
    return Point(x / a, y / a);
  }

  double abs2() const {
    return x * x + y * y;
  }

  double abs() const {
    return ::sqrt(abs2());
  }
};

}

#endif
