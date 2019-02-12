#ifndef __LAYOUT_HH__
#define __LAYOUT_HH__ 1

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "random.hh"
#include "persist.hh"

namespace makemore {

struct Layout : Persist {
  unsigned int n;
  double *x, *y, *r;

  Layout() {
    n = 0;
    x = y = r = NULL;
  }
 
  Layout(const Layout &lay) {
    n = lay.n;

    x = new double[n];
    y = new double[n];
    r = new double[n];

    memcpy(x, lay.x, n * sizeof(double));
    memcpy(y, lay.y, n * sizeof(double));
    memcpy(r, lay.r, n * sizeof(double));
  }

  Layout(unsigned int n);
  ~Layout();

  static Layout *new_square_grid(unsigned int dim, double s = 1.0, unsigned int chan = 1);
  static Layout *new_square_random(unsigned int n, double s = 1.0);
  static Layout *new_square_center(unsigned int n, double s = 1.0);
  static Layout *new_text(unsigned int bits = 256, unsigned int chan = 4);
  static Layout *new_line(unsigned int dim, double s = 1.0, unsigned int chan = 1);

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;

  Layout &operator +=(const Layout &x);
};

}

#endif
