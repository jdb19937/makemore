#ifndef __MAKEMORE_TRON_HH__
#define __MAKEMORE_TRON_HH__ 1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

struct Tron {
  double errdecay;

  unsigned int inn, outn;

  double cerr2, cerrm;

  Tron() {
    inn = 0;
    outn = 0;
    cerr2 = 0.5;
    cerrm = 0.5;
    errdecay = 0.001;
  }

  Tron(unsigned int _inn, unsigned int _outn) : inn(_inn), outn(_outn) {
    cerr2 = 0.5;
    cerrm = 0.5;
    errdecay = 0.001;
  }

  virtual const double *feed(const double *in, double *fin) = 0;

  const double *feed(const double *in) {
    return feed(in, NULL);
  }

  virtual void train(double r) = 0;

  inline void train() {
    train(0.001);
  }

  void target(const double *tgt);

  virtual const double *input() = 0;
  virtual const double *finput() = 0;
  virtual const double *output() = 0;
  virtual double *foutput() = 0;

  virtual void sync(double t) { }
};

struct Compositron : Tron {
  Tron *a, *b;

  Compositron(Tron *_a, Tron *_b) : Tron(_a->inn, _b->outn) {
    a = _a;
    b = _b;
  }

  virtual const double *feed(const double *in, double *fin) {
    const double *aout = a->feed(in, fin);
    return b->feed(aout, a->foutput());
  }

  virtual void train(double r) {
    b->train(r);
    a->train(r);
  }

  virtual const double *output() { return b->output(); }
  virtual const double *input() { return a->input(); }
  virtual const double *finput() { return a->finput(); }
  virtual double *foutput() { return b->foutput(); }

  virtual void target(const double *tgt) {
    b->target(tgt);
  }

  virtual void sync(double t) {
    a->sync(t);
    b->sync(t);
  }
};

inline Compositron *compositron(Tron *f, Tron *g) {
  return new Compositron(f, g);
}



struct Passthrutron : Tron {
  Tron *t;
  unsigned int mbn;
  unsigned int k;
  unsigned int inrn, outrn;

  double *out, *fout;

  Passthrutron(unsigned int k, unsigned int _mbn, Tron *_t);
  virtual ~Passthrutron();

  virtual const double *input() { return t->input(); }
  virtual const double *output() { return out; }
  virtual const double *finput() { return t->finput(); }
  virtual double *foutput() { return fout; }

  virtual const double *feed(const double *in, double *fin);
  virtual void train(double nu);

  virtual void target(const double *tgt) {
    t->target(tgt);
  }
};

inline Passthrutron *passthrutron(unsigned int k, unsigned int mbn, Tron *t) {
  return new Passthrutron(k, mbn, t);
}

#if 0
struct Extron : Tron {
  Tron *t;

  const double *in;
  double *fin;

  Extron(unsigned int n, Tron *_t) : Tron(n, n) {
    t = _t;
    assert(inn >= t->inn);
  }

  virtual ~Extron() {
  }

  virtual const double *input() { return in; }
  virtual const double *output() { return in; }
  virtual const double *finput() { return fin; }
  virtual double *foutput() { return fin; }

  virtual const double *feed(const double *_in, double *_fin) {
    in = _in;
    fin = _fin;

    if (fin)
      t->feed(in + inn - t->inn, fin + inn - t->inn);
    else
      t->feed(in + inn - t->inn, NULL);

    return in;
  }

  virtual void train(double r) {

  }
};

inline Extron *extron(unsigned int n, Tron *t) {
  return new Extron(n, t);
}
#endif

struct Identron : Tron {
  const double *in;
  double *fin;

  Identron(unsigned int n) : Tron(n, n) { }

  virtual const double *feed(const double *_in, double *_fin) {
    in = _in;
    fin = _fin;
    return in;
  }

  virtual void train(double r) { }

  virtual const double *input() { return in; }
  virtual const double *output() { return in; }
  virtual const double *finput() { return fin; }
  virtual double *foutput() { return fin; }
};

inline Identron *identron(unsigned int n) {
  return new Identron(n);
}

#endif
