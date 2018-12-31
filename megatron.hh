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
  unsigned int *wimap, *womap;

  double *weight;

  unsigned int inrn, outrn;
  unsigned int mbn;

  double eta;
  bool activated;

  double *cweight;

  std::vector<std::vector<unsigned int> > _mow;

  Megatron(const Wiring *_wire, double *_cweight, unsigned int _mbn = 1, double _eta = 1.0, bool _activated = true);
  virtual ~Megatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *output() { return out; }
  virtual const double *input() { return in; }
  virtual double *foutput() { return fout; }

  virtual void sync(double t);


  void _makemaps();
  void randomize(double disp);
};

#endif
