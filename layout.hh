#ifndef __LAYOUT_HH__
#define __LAYOUT_HH__ 1

#include <stdio.h>
#include <string.h>
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
 
  Layout(const Layout &lay) {
    n = lay.n;

    if (x) delete[] x;
    if (y) delete[] y;
    if (r) delete[] r;

    x = new double[n];
    y = new double[n];
    r = new double[n];

    memcpy(x, lay.x, n * sizeof(double));
    memcpy(y, lay.y, n * sizeof(double));
    memcpy(r, lay.r, n * sizeof(double));
  }

  Layout(unsigned int n);
  ~Layout();

  static Layout *new_square_grid(unsigned int dim, double s = 1.0);
  static Layout *new_square_random(unsigned int n, double s = 1.0);

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;

  Layout &operator +=(const Layout &x);
};

#endif
