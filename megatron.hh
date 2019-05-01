#ifndef __MAKEMORE_MEGATRON_HH__
#define __MAKEMORE_MEGATRON_HH__ 1

#include "tron.hh"
#include "wiring.hh"

namespace makemore {

extern double adam_b1, adam_b2, adam_b3, adam_eps;

struct Megatron : Tron {
  const double *in;
  double *fin, *out, *fout;

  const class Wiring *wire;
  class Mapfile *mapfile;

  unsigned int wn;
  unsigned int **iwmap, **owmap;
  unsigned int **iomap, **oimap;
  unsigned int *wimap, *womap;

  double *weight;
  unsigned int *mapbuf;

double *m, *v;

  unsigned int inrn, outrn;
  unsigned int mbn;

  double eta;
  bool activated;

  std::vector<std::vector<unsigned int> > _mow;

  Megatron(const class Wiring *_wire, class Mapfile *_mapfile, unsigned int _mbn = 1, double _eta = 1.0, bool _activated = true);
  virtual ~Megatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *output() { return out; }
  virtual const double *input() { return in; }
  virtual double *foutput() { return fout; }

  void _makemaps();
  void randomize(double disp);
};

}

#endif
