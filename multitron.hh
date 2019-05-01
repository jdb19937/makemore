#ifndef __MAKEMORE_MULTITRON_HH__
#define __MAKEMORE_MULTITRON_HH__ 1

#include "tron.hh"
#include "layout.hh"
#include "wiring.hh"
#include "megatron.hh"
#include "topology.hh"

#include <vector>

namespace makemore {

struct Multitron : Tron {
  unsigned int mbn;
  unsigned int inrn, outrn;

  std::vector<Tron*> trons;
  Tron *mt0, *mt1;

  class Mapfile *mapfile;

  Multitron(const Topology &top, class Mapfile *_mapfile, unsigned int _mbn = 1, bool activated = false, bool normalized = false);
  virtual ~Multitron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *input()
    { return mt0->input(); }
  virtual const double *output()
    { return mt1->output(); }
  virtual double *foutput()
    { return mt1->foutput(); }

  void _makemaps();
  virtual void randomize(double dispersion);
};

}

#endif
