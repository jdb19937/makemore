#ifndef __MAKEMORE_NORMATRON_HH__
#define __MAKEMORE_NORMATRON_HH__ 1

#include "tron.hh"
#include "wiring.hh"
#include "mapfile.hh"

namespace makemore {

struct Normatron : Tron {
  unsigned int mbn, n;
  double beta;

  Mapfile *mapfile;
  double *mean, *var;

  const double *in;
  double *fin;

  double *out;
  double *fout;

  Normatron(class Mapfile *_mapfile, unsigned int _n, unsigned int _mbn, double _beta = 0.1);
  virtual ~Normatron();

  virtual const double *feed(const double *_in, double *_fin);

  virtual void train(double nu);

  virtual const double *input() { return in; }
  virtual const double *output() { return out; }
  virtual double *foutput() { return fout; }

  virtual void randomize(double disp);
};

}

#endif
