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

  double err2, errm;
  unsigned int rounds;

  Tron() {
    inn = 0;
    outn = 0;
    err2 = 0;
    errm = 0;
    errdecay = 0.001;
    rounds = 0;
  }

  Tron(unsigned int _inn, unsigned int _outn) : inn(_inn), outn(_outn) {
    err2 = 0;
    errm = 0;
    errdecay = 0.001;
    rounds = 0;
  }

  virtual const double *feed(const double *in, double *fin) = 0;

  const double *feed(const double *in) {
    return feed(in, NULL);
  }

  virtual void train(double r) = 0;

  inline void train() {
    train(0.001);
  }

  virtual void target(const double *tgt, bool update_stats = true);

  virtual const double *input() = 0;
  virtual const double *output() = 0;
  virtual double *foutput() = 0;

  virtual void sync(double t) { }
};

struct Compositron : Tron {
  Tron *a, *b;

  Compositron(Tron *_a, Tron *_b) : Tron(_a->inn, _b->outn) {
    a = _a;
    b = _b;

    assert(a->outn == b->inn);
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
  virtual double *foutput() { return b->foutput(); }

  virtual void sync(double t) {
    a->sync(t);
    b->sync(t);
  }
};

inline Compositron *compositron(Tron *f, Tron *g) {
  return new Compositron(f, g);
}


struct Twiddletron : Tron {
  Tron *t;
  unsigned int mbn;
  unsigned int twoff;
  unsigned int twlen;
  unsigned int dim;

  unsigned int inrn, outrn;
  double *out, *fout;

  double *buf1, *buf2;

  Twiddletron(Tron *_t, unsigned int _mbn, unsigned int _twoff, unsigned int _twlen);
  virtual ~Twiddletron();

  virtual const double *input() { return t->input(); }
  virtual const double *output() { return out; }
  virtual double *foutput() { return fout; }

  virtual const double *feed(const double *in, double *fin);
  virtual void train(double nu);
};


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
  virtual double *foutput() { return fout; }

  virtual const double *feed(const double *in, double *fin);
  virtual void train(double nu);
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
  virtual double *foutput() { return fin; }
};

inline Identron *identron(unsigned int n) {
  return new Identron(n);
}

#endif
