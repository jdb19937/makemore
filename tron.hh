#ifndef __MAKEMORE_TRON_HH__
#define __MAKEMORE_TRON_HH__ 1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

struct Tron {
  unsigned int inn, outn;

  double cerr1, cerr2, cerr3;

  Tron(unsigned int _inn, unsigned int _outn) : inn(_inn), outn(_outn) {
    cerr1 = 0.5;
    cerr2 = 0.5;
    cerr3 = 0.5;
  }

  virtual const double *feed(const double *in, double *fin) = 0;

  const double *feed(const double *in) {
    return feed(in, NULL);
  }

  virtual void train(double r) = 0;

  inline void train() {
    train(0.001);
  }

  void target(const double *tgt) {
    double *fout = foutput();
    const double *out = output();

    for (unsigned int i = 0; i < outn; ++i)
      fout[i] = tgt[i] - out[i];

    cerr1 *= 0.999;
    cerr1 += 0.001 * err1();

    cerr2 *= 0.999;
    cerr2 += 0.001 * err2();

    cerr3 *= 0.999;
    cerr3 += 0.001 * err3();
  }

  void target(bool n) {
    assert(outn == 1);
    double tgt[1];
    tgt[0] = n ? 1.0 : 0.0;
    target(tgt);
  }

  void target(double x) {
    assert(outn == 1);
    double tgt[1];
    tgt[0] = x;
    target(tgt);
  }

  void report() {
    fprintf(stderr, "cerr1=%lf cerr2=%lf cerr3=%lf\n", cerr1, cerr2, cerr3);
  }

  virtual const double *input() = 0;
  virtual const double *finput() = 0;
  virtual const double *output() = 0;
  virtual double *foutput() = 0;

  virtual void sync() { }

  double err3();
  double err2();
  double err1();
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

  virtual void sync() {
    a->sync();
    b->sync();
  }
};

inline Compositron *compositron(Tron *f, Tron *g) {
  return new Compositron(f, g);
}


struct Encudatron : Tron {
  const double *in;
  double *fin;
  double *out;
  double *fout;

  Encudatron(unsigned int n);
  ~Encudatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *input() { return in; }
  virtual const double *output() { return out; }
  virtual const double *finput() { return fin; }
  virtual double *foutput() { return fout; }
};

inline Encudatron *encudatron(unsigned int n) {
  return new Encudatron(n);
}

struct Decudatron : Tron {
  const double *in;
  double *fin, *dfin_tmp;
  double *out;
  double *fout;

  Decudatron(unsigned int n);
  ~Decudatron();

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);

  virtual const double *input() { return in; }
  virtual const double *output() { return out; }
  virtual const double *finput() { return fin; }
  virtual double *foutput() { return fout; }
};

inline Decudatron *decudatron(unsigned int n) {
  return new Decudatron(n);
}


struct Intron : Tron {
  Tron *t;

  const double *in;
  double *fin;
  double *out, *fout;

  Intron(unsigned int n, Tron *_t);
  virtual ~Intron();

  virtual const double *input() { return in; }
  virtual const double *output() { return out; }
  virtual const double *finput() { return fin; }
  virtual double *foutput() { return fout; }

  virtual const double *feed(const double *_in, double *_fin);
  virtual void train(double r);
};

inline Intron *intron(unsigned int n, Tron *t) {
  return new Intron(n, t);
}

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
