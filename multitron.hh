#ifndef __MAKEMORE_MULTITRON_HH__
#define __MAKEMORE_MULTITRON_HH__ 1

#include "tron.hh"
#include "layout.hh"
#include "wiring.hh"
#include "megatron.hh"

#include <vector>

struct Multitron : Tron {
  unsigned int mbn;
  unsigned int inrn, outrn;

  std::vector<Megatron*> megatrons;
  Megatron *mt0, *mt1;

  int fd;
  const char *fn;
  size_t map_size;
  double *map;

  Multitron(const Topology &top, unsigned int _mbn = 1, const char *fn = NULL);
  virtual ~Multitron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *input()
    { return mt0->input(); }
  virtual const double *output()
    { return mt1->output(); }
  virtual double *foutput()
    { return mt1->foutput(); }

  virtual void sync(double t);


  void _makemaps();
  void randomize(double disp = 4.0);
};

#endif
