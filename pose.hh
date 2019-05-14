#ifndef __MAKEMORE_POSE_HH__
#define __MAKEMORE_POSE_HH__ 1

#include <vector>
#include <string>
#include <set>

#include "point.hh"
#include "triangle.hh"

namespace makemore {

struct Pose {
  Point center;
  double scale, angle, stretch, skew;

  Pose() {
    center.x = 0;
    center.y = 0;
    scale = 0;
    angle = 0;
    stretch = 0;
    skew = 0;
  }

  Pose(double centerx, double centery, double _scale, double _angle = 0.0, double _stretch = 1.0, double _skew = 0.0) {
    center.x = centerx;
    center.y = centery;
    scale = _scale;
    angle = _angle;
    stretch = _stretch;
    skew = _skew;
  }

  Pose(const Pose &pose) {
    center = pose.center;
    scale = pose.scale;
    angle = pose.angle;
    stretch = pose.stretch;
    skew = pose.skew;
  }

  Pose(const Triangle &tri);

  ~Pose() {
  }

  Triangle mark() const;
};

}

#endif
