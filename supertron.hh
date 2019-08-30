#ifndef __MAKEMORE_SUPERTRON_HH__
#define __MAKEMORE_SUPERTRON_HH__ 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mapfile.hh"

namespace makemore {

struct Supertron {
  class Mapfile *mapfile;

  double err2, errm, errdecay;
  unsigned int rounds;

  unsigned int inn, outn;

  struct Layer {
    Layer() {
      head = NULL;
      in = NULL;
      fin = NULL;
      out = NULL;
      fout = NULL;
      weight = NULL;
      bias = NULL;
      mbias = NULL; 
      vbias = NULL;
      m = NULL;
      v = NULL;

      wn = 0;
      inn = 0;
      outn = 0;
    }

    ~Layer() {
    }

    Layer(const Layer &l) {
      head = l.head;
      in = l.in;
      fin = l.fin;
      out = l.out;
      fout = l.fout;
      weight = l.weight;
      bias = l.bias;
      mbias = l.mbias;
      vbias = l.vbias;
      m = l.m;
      v = l.v;

      wn = l.wn;
      inn = l.inn;
      outn = l.outn;
    }

    enum Type {
      TYPE_FULL = 0,
      TYPE_LOCAL = 1,
      TYPE_CONV = 2,
      TYPE_YCONV = 3
    };

    struct Head {
      Layer::Type type;
      unsigned int wn, inn, outn;

      unsigned int iw, ih, ic;
      unsigned int ow, oh, oc;
      unsigned int d;
      int s;

      double adam_b1, adam_b2, adam_b3, adam_eps, eta;
      bool activated;
    };

    Head *head;
    const double *in;
    double *fin, *out, *fout;
    double *weight, *m, *v;
    double *bias, *mbias, *vbias;

    unsigned int wn, inn, outn;
  };

  uint64_t *cunlayers;
  std::vector<Layer*> layers;
  Layer *first_layer, *last_layer;

  Supertron(class Mapfile *_mapfile);
  ~Supertron();

  const double *feed(const double *_in, double *_fin = NULL);
  void train(double r);
  void target(const double *_tgt);
  void update_stats();

  const double *input() { assert(layers.size()); return layers[0]->in; }
  const double *output() { assert(layers.size()); return layers[layers.size() - 1]->out; }
  double *foutput() { assert(layers.size()); return layers[layers.size() - 1]->fout; }

  void randomize(double disp);

  void add_layer(const Layer::Head &head);
  void add_layer(
    Layer::Type type,
    unsigned int iw, unsigned int ih, unsigned int ic,
    unsigned int ow, unsigned int oh, unsigned int oc,
    unsigned int d, int s,
    bool activated
  );
};

}

#endif
