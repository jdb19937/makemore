#include <math.h>

#include "layout.hh"
  
Layout::Layout(unsigned int _n) {
  n = _n;
  x = new double[n]();
  y = new double[n]();
  r = new double[n]();

  for (unsigned int i = 0; i < n; ++i) {
    x[i] = 0.5;
    y[i] = 0.5;
    r[i] = 0.5;
  }
}

Layout::~Layout() {
  delete[] x;
  delete[] y;
  delete[] r;
}

Layout *Layout::new_square_grid(unsigned int dim, double s) {
  Layout *l = new Layout(dim * dim);

  double pi = atan2(0, -1);
  double r = sqrt(s / pi) / (double)dim;

  for (unsigned int j = 0; j < dim; ++j) {
    for (unsigned int i = 0; i < dim; ++i) {
      l->x[j * dim + i] = ((double)i + 0.5) / (double)dim;
      l->y[j * dim + i] = ((double)j + 0.5) / (double)dim;
      l->r[j + dim + i] = r;
    }
  }

  return l;
}

Layout *Layout::new_square_random(unsigned int n, double s) {
  Layout *l = new Layout(n);

  double pi = atan2(0, -1);
  double r = sqrt(s / (pi * (double)n));

  for (unsigned int i = 0; i < l->n; ++i) {
    l->x[i] = rnd();
    l->y[i] = rnd();
    l->r[i] = r;
  }

  return l;
}

