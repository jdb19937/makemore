#ifndef __MAKEMORE_SUPERTRON_HH__
#define __MAKEMORE_SUPERTRON_HH__ 1

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

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
      pout = NULL;
      weight = NULL;
      wbuf = NULL;
      wbufk = 0;
      m = NULL;
      v = NULL;
      noise = NULL;

      wn = 0;
      inn = 0;
      outn = 0;

      oiwtab = NULL;
      iowtab = NULL;
      wiotab = NULL;
    }

    ~Layer() {
    }

    Layer(const Layer &l) {
      head = l.head;
      in = l.in;
      fin = l.fin;
      out = l.out;
      fout = l.fout;
      pout = l.pout;
      weight = l.weight;
      wbuf = l.wbuf;
      wbufk = l.wbufk;
      m = l.m;
      v = l.v;

      wn = l.wn;
      inn = l.inn;
      outn = l.outn;

      oiwtab = l.oiwtab;
      iowtab = l.iowtab;
      wiotab = l.wiotab;

      type = l.type;
      noise = l.noise;
    }

    enum Type {
      TYPE_FULL = 0,
      TYPE_LOCAL = 1,
      TYPE_CONV = 2,
      TYPE_YCONV = 3,
      TYPE_MPOOL = 4,
      TYPE_IDENTITY = 5,
      TYPE_NOISEPAD = 6,
      TYPE_FILLPAD = 7,
      TYPE_PADINPUT = 8,
      TYPE_ADDINPUT = 9
    };

    enum Activation {
      ACTIVATION_NONE = 0,
      ACTIVATION_SIGMOID = 1,
      ACTIVATION_RELU = 2,
      ACTIVATION_SOFTPLUS = 3,
      ACTIVATION_ABS = 4,
      ACTIVATION_SQUARE = 5
    };

    struct Head {
      Layer::Type type;
      unsigned int wn, inn, outn;

      unsigned int iw, ih, ic;
      unsigned int ow, oh, oc;
      unsigned int d;
      int s;

      bool adam;
      double adam_b1, adam_b2, adam_b3, adam_eps, eta;

      int8_t activated;
    };

    Head *head;
    const double *in;
    double *fin, *out, *pout, *fout;
    double *weight, *m, *v;

    double *wbuf;
    unsigned int wbufk;

    unsigned int wn, inn, outn;

    int *oiwtab, *iowtab, *wiotab;

    double *noise;
    Layer::Type type;
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
  double *output() { assert(layers.size()); return layers[layers.size() - 1]->out; }
  double *foutput() { assert(layers.size()); return layers[layers.size() - 1]->fout; }

  void randomize_last(double disp);
  void randomize(double disp);
  void randomize_layer(Layer *, double disp);

  void add_layer(const Layer::Head &head);
  void add_layer(
    Layer::Type type,
    unsigned int iw, unsigned int ih, unsigned int ic,
    unsigned int ow, unsigned int oh, unsigned int oc,
    unsigned int d, int s,
    Layer::Activation activated
  );
};

}

#endif
