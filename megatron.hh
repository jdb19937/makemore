#ifndef __MAKEMORE_MEGATRON_HH__
#define __MAKEMORE_MEGATRON_HH__ 1

#include "tron.hh"
#include "layout.hh"
#include "wiring.hh"

struct Megatron : Tron {
  const double *in;
  double *fin, *out, *fout;

  const Wiring *wire;

  unsigned int wn;
  unsigned int **iwmap, **owmap;
  unsigned int **iomap, **oimap;

  double *weight;

  unsigned int inrn, outrn;
  unsigned int mbn;

  double eta, kappa;

  double *cweight;

  Megatron(const Wiring *_wire, double *_cweight, unsigned int _mbn = 1);
  virtual ~Megatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *finput() { return fin; }
  virtual const double *output() { return out; }
  virtual const double *input() { return in; }
  virtual double *foutput() { return fout; }

  virtual void sync();


  void _makemaps();
  void randomize(double disp = 4.0);
};

#endif
