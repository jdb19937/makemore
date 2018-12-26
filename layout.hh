#ifndef __LAYOUT_HH__
#define __LAYOUT_HH__ 1

#include "random.hh"

struct Layout {
  unsigned int n;
  double *x, *y, *r;

  Layout(unsigned int n);
  ~Layout();

  static Layout *new_square_grid(unsigned int dim, double s = 0.0);
  static Layout *new_square_random(unsigned int n, double s = 0.0);
  static Layout *new_square_random2(unsigned int n, double s = 0.0);
};

#endif
