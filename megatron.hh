#ifndef __MEGATRON_HH__
#define __MEGATRON_HH__ 1

#include "tron.hh"
#include "layout.hh"

struct Megatron : Tron {
  const double *in;
  double *fin, *out, *fout;

  const Layout *inl, *outl;

  unsigned int wn;
  unsigned int **iwmap, **owmap;
  unsigned int **iomap, **oimap;

  double *weight;

  double eta, kappa;

  Megatron(const Layout *_inl, const Layout *_outl);
  virtual ~Megatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *finput() { return fin; }
  virtual const double *output() { return out; }
  virtual const double *input() { return in; }
  virtual double *foutput() { return fout; }


  void makemaps(unsigned int minv = 0, unsigned int maxv = (1<<16), double disp = 4.0);
};

#endif
