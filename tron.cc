#define __MAKEMORE_TRON_CC__ 1

#include <stdio.h>

#include <math.h>

#include "tron.hh"
#include "random.hh"
#include "cudamem.hh"

void Tron::target(const double *tgt, bool update_stats) {
  const double *out = output();
  double *fout = foutput();

  cusubvec(tgt, out, outn, fout);

  if (update_stats) {
    double z = pow(1.0 - errdecay, (double)rounds);

    double nerr2 = sqrt(cusumsq(fout, outn) / outn);
    err2 *= (1.0 - z);
    err2 *= (1.0 - errdecay);
    err2 += errdecay * nerr2;
    err2 *= 1.0 / (1.0 - z * (1.0 - errdecay));

    double nerrm = cumaxabs(fout, outn);
    errm *= (1.0 - z);
    errm *= (1.0 - errdecay);
    errm += errdecay * nerrm;
    errm *= 1.0 / (1.0 - z * (1.0 - errdecay));

    ++rounds;
  }
}

Passthrutron::Passthrutron(unsigned int _k, unsigned int _mbn, Tron *_t) {
  err2 = 0;
  errm = 0;

  k = _k;
  mbn = _mbn;
  t = _t;

  assert(t->inn % mbn == 0);
  inrn = t->inn / mbn;
  assert(inrn >= k);
  assert(t->outn % mbn == 0);
  outrn = t->outn / mbn;

  inn = t->inn;

  outrn += k;
  outn = mbn * outrn;

  cumake(&out, outn);
  cumake(&fout, outn);
}

Passthrutron::~Passthrutron() {
  cufree(out);
  cufree(fout);
}

const double *Passthrutron::feed(const double *in, double *fin) {
  const double *tout = t->feed(in, fin);

  cuzero(fout, outn);
  cucutpaste(in, tout, mbn, inrn, outrn - k, outrn, out);

  return out;
}

void Passthrutron::train(double nu) {
  if (double *tfout = t->foutput())
    cucutadd(fout, mbn, outrn, outrn - k, tfout);
  t->train(nu);
}

#if TRONTEST_MAIN
int main() {
  Tron *i = new Identron(2);
  Tron *j = new Encudatron(2);
  Tron *k = new Decudatron(2);
  Tron *jk = new Compositron(j, k);
  Tron *ijk = new Compositron(i, jk);
  
  double x[2] = {1,5};
  const double *y = ijk->feed(x);
  printf("%lf %lf\n", y[0], y[1]);
  return 0;
}
#endif
