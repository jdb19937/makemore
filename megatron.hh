#ifndef __MAKEMORE_MEGATRON_HH__
#define __MAKEMORE_MEGATRON_HH__ 1

#include "tron.hh"
#include "layout.hh"
#include "wiring.hh"

struct Megatron : Tron {
  const double *in;
  double *fin, *out, *fout;

  const Wiring *wire;
  const Layout *inl, *outl;

  unsigned int wn;
  unsigned int **iwmap, **owmap;
  unsigned int **iomap, **oimap;

  double *weight;

  unsigned int inrn, outrn;
  unsigned int mbn;

  double eta, kappa;

  Megatron(const Wiring *_wire, unsigned int _mbn = 1);
  virtual ~Megatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *finput() { return fin; }
  virtual const double *output() { return out; }
  virtual const double *input() { return in; }
  virtual double *foutput() { return fout; }


  void _makemaps(double disp = 4.0);

  virtual void load(FILE *fp) { assert(!"todo"); }
  virtual void save(FILE *fp) const { assert(!"todo"); }
};

#endif
