#define __MAKEMORE_TRON_CC__ 1

#include <stdio.h>

#include <math.h>

#include "tron.hh"
#include "random.hh"
#include "cudamem.hh"

void Tron::target(const double *tgt) {
  const double *out = output();
  double *fout = foutput();

  cusubvec(tgt, out, outn, fout);

  double err2 = sqrt(cusumsq(fout, outn) / outn);
  cerr2 *= (1.0 - errdecay);
  cerr2 += errdecay * err2;

  double errm = cumaxabs(fout, outn);
  cerrm *= (1.0 - errdecay);
  cerrm += errdecay * errm;
}

#if 0
double Tron::err3() {
  double *fout = foutput();
  if (!fout)
    return 0;

  double me = 0;
  for (unsigned int i = 0; i < outn; ++i) {
    double e = fout[i] * fout[i];
    if (e > me)
      me = e;
  }
  me = sqrt(me);
  return me;
}

double Tron::err2() {
  double *fout = foutput();
  if (!fout)
    return 0;

  double e = 0;
  for (unsigned int i = 0; i < outn; ++i)
    e += fout[i] * fout[i];
  e /= (double)outn;
  e = sqrt(e);
  return e;
}

double Tron::err1() {
  double *fout = foutput();
  if (!fout)
    return 0;

  double e = 0;
  for (unsigned int i = 0; i < outn; ++i)
    e += (abs(fout[i]) < 0.5) ? 0.0 : 1.0;
  e /= (double)outn;
  return e;
}
#endif


Passthrutron::Passthrutron(unsigned int _k, unsigned int _mbn, Tron *_t) {
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

  out = cunew<double>(outn);
  fout = cunew<double>(outn);
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
  cucutadd(fout, mbn, outrn, outrn - k, t->foutput());
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
