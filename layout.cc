#include <stdint.h>
#include <string.h>
#include <netinet/in.h>

#include <math.h>

#include "layout.hh"

namespace makemore {
  
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

Layout *Layout::new_square_grid(unsigned int dim, double s, unsigned int chan) {
  Layout *l = new Layout(dim * dim * chan);

  double pi = atan2(0, -1);
  double r = sqrt(s / pi) / (double)dim;

  for (unsigned int j = 0; j < dim; ++j) {
    for (unsigned int i = 0; i < dim; ++i) {
      for (unsigned int c = 0; c < chan; ++c) {
        l->x[chan * (j * dim + i) + c] = ((double)i + 0.5) / (double)dim;
        l->y[chan * (j * dim + i) + c] = ((double)j + 0.5) / (double)dim;
        l->r[chan * (j + dim + i) + c] = r;
      }
    }
  }

  return l;
}

Layout *Layout::new_line(unsigned int dim, double s, unsigned int chan) {
  Layout *l = new Layout(dim * chan);

  double r = s / (double)dim;
  for (unsigned int i = 0; i < dim; ++i) {
    for (unsigned int c = 0; c < chan; ++c) {
      l->x[chan * i + c] = ((double)i + 0.5) / (double)dim;
      l->y[chan * i + c] = 0.5;
      l->r[chan * i + c] = r;
    }
  }

  return l;
}

Layout *Layout::new_text(unsigned int bits, unsigned int chan) {
  Layout *l = new Layout(bits * chan);

  for (unsigned int c = 0; c < chan; ++c) {
    for (unsigned int i = 0; i < bits; ++i) {
      l->x[bits * c + i] = ((double)i + 0.5) / (double)bits;
      l->y[bits * c + i] = 0.5;
      l->r[bits * c + i] = 0;
    }
  }

  return l;
}

Layout *Layout::new_square_random(unsigned int n, double s) {
  Layout *l = new Layout(n);

  double pi = atan2(0, -1);
  double r = sqrt(s / (pi * (double)n));

  for (unsigned int i = 0; i < l->n; ++i) {
    l->x[i] = randrange(0.0, 1.0);
    l->y[i] = randrange(0.0, 1.0);
    l->r[i] = r;
  }

  return l;
}

Layout *Layout::new_square_center(unsigned int n, double r) {
  Layout *l = new Layout(n);

  double pi = atan2(0, -1);

  for (unsigned int i = 0; i < l->n; ++i) {
    l->x[i] = 0.5;
    l->y[i] = 0.5;
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

Layout &Layout::operator +=(const Layout &lay) {
  double *nx = new double[n + lay.n];
  memcpy(nx, x, n * sizeof(double));
  memcpy(nx + n, lay.x, lay.n * sizeof(double));

  double *ny = new double[n + lay.n];
  memcpy(ny, y, n * sizeof(double));
  memcpy(ny + n, lay.y, lay.n * sizeof(double));

  double *nr = new double[n + lay.n];
  memcpy(nr, r, n * sizeof(double));
  memcpy(nr + n, lay.r, lay.n * sizeof(double));

  n += lay.n;
  if (x) delete[] x;
  x = nx;
  if (y) delete[] y;
  y = ny;
  if (r) delete[] r;
  r = nr;

  return *this;
}

}
