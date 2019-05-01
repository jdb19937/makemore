#ifndef __MAKEMORE_CHOLO_HH__
#define __MAKEMORE_CHOLO_HH__ 1

#include <string>

namespace makemore {

struct Cholo {
  double *tmp;

  double *in, *out, *mean, *var, *cov, *chol;
  double *unchol;

  unsigned int ichol;
  unsigned int dim2;

  void reset();
  void finalize();

  const unsigned int dim;

  Cholo(unsigned int _dim);
  ~Cholo();

  void observe(const double *x);
  void observecu(const double *x);
  void generate(double *x, double m = 1.0);
  void generate(const double *y, double *x);
  void encode(const double *x, double *y);

  void save(FILE *fp);
  void load(FILE *fp);
  void load(const std::string &fn);
};

}

#endif
