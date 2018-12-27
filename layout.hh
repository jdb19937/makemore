#ifndef __LAYOUT_HH__
#define __LAYOUT_HH__ 1

#include <assert.h>

#include "random.hh"
#include "persist.hh"

struct Layout : Persist {
  unsigned int n;
  double *x, *y, *r;

  Layout() {
    n = 0;
    x = y = r = NULL;
  }

  Layout(unsigned int n);
  ~Layout();

  static Layout *new_square_grid(unsigned int dim, double s = 1.0);
  static Layout *new_square_random(unsigned int n, double s = 1.0);

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

#endif
