#include "layout.hh"
  
Layout::Layout(unsigned int _n) {
  n = _n;
  x = new double[n]();
  y = new double[n]();
  r = new double[n]();

  for (unsigned int i = 0; i < n; ++i) {
    x[i] = 0.5;
    y[i] = 0.5;
  }
}

Layout::~Layout() {
  delete[] x;
  delete[] y;
  delete[] r;
}

Layout *Layout::new_square_grid(unsigned int dim, double s) {
  Layout *l = new Layout(dim * dim);

  for (unsigned int j = 0; j < dim; ++j) {
    for (unsigned int i = 0; i < dim; ++i) {
      l->x[j * dim + i] = ((double)i + 0.5) / (double)dim;
      l->y[j * dim + i] = ((double)j + 0.5) / (double)dim;
      l->r[j + dim + i] = s;
    }
  }

  return l;
}

Layout *Layout::new_square_random(unsigned int n, double s) {
  Layout *l = new Layout(n);

  for (unsigned int i = 0; i < l->n; ++i) {
    l->x[i] = rnd();
    l->y[i] = rnd();
    l->r[i] = s;
  }

  return l;
}

Layout *Layout::new_square_random2(unsigned int n, double s) {
  Layout *l = new Layout(n);

  for (unsigned int i = 0; i < l->n; ++i) {
    l->x[i] = rnd() < 0.5 ? 0.5 - rnd() * rnd() : 0.5 + rnd() * rnd();
    l->y[i] = rnd() < 0.5 ? 0.5 - rnd() * rnd() : 0.5 + rnd() * rnd();
    l->r[i] = s;
  }

  return l;
}
