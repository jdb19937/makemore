#include <stdint.h>
#include <netinet/in.h>

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

void Layout::save(FILE *fp) const {
  uint32_t on = htonl(n);
  assert(1 == fwrite(&on, 4, 1, fp));
  assert(n == fwrite(x, sizeof(double), n, fp));
  assert(n == fwrite(y, sizeof(double), n, fp));
  assert(n == fwrite(r, sizeof(double), n, fp));
}

void Layout::load(FILE *fp) {
  uint32_t on;
  assert(1 == fread(&on, 4, 1, fp));
  n = ntohl(on);
  
  x = new double[n];
  assert(n == fread(x, sizeof(double), n, fp));

  y = new double[n];
  assert(n == fread(y, sizeof(double), n, fp));

  r = new double[n];
  assert(n == fread(r, sizeof(double), n, fp));
}
