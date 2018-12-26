#include <stdio.h>

#include <math.h>

#include "tron.hh"
#include "random.hh"
#include "cudamem.hh"

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

Encudatron::Encudatron(unsigned int n) : Tron(n, n) {
  out = cumake(outn);
  fout = cumake(outn);
}

Encudatron::~Encudatron() {
  cufree(out);
  cufree(fout);
}

const double *Encudatron::feed(const double *_in, double *_fin) {
  assert(inn == outn);

  in = _in;
  fin = _fin;

  encude(in, inn, out);
  cuzero(fout, outn);

  return out;
}

void Encudatron::train(double r) {
  assert(inn == outn);

  if (!fin)
    return;

  double dfin[inn];
  decude(fout, outn, dfin);

  for (unsigned int i = 0; i < inn; ++i)
    fin[i] += dfin[i];
}

Decudatron::Decudatron(unsigned int n) : Tron(n, n) {
  out = new double[outn]();
  fout = new double[outn]();
  dfin_tmp = cumake(inn);
}

Decudatron::~Decudatron() {
  delete[] out;
  delete[] fout;
  cufree(dfin_tmp);
}

const double *Decudatron::feed(const double *_in, double *_fin) {
  assert(inn == outn);

  in = _in;
  fin = _fin;

  decude(in, inn, out);
  memset(fout, 0, outn * sizeof(double));

  return out;
}

void Decudatron::train(double r) {
  assert(inn == outn);

  if (fin) {
    encude(fout, outn, dfin_tmp);
    cuaddvec(fin, dfin_tmp, inn, fin);
  }
}

Intron::Intron(unsigned int n, Tron *_t) : Tron(n, _t->outn + n) {
  t = _t;
  out = cumake(outn);
  fout = cumake(outn);
}

Intron::~Intron() {
  cufree(out);
  cufree(fout);
}

const double *Intron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  cucopy(in, inn, out);
  if (t->outn > 0)
    cucopy(t->output(), t->outn, out + inn);
  cuzero(fout, outn);

  return out;
}

void Intron::train(double r) {
  double *tfout = t->foutput();

  if (tfout && t->outn > 0)
    cuaddvec(fout + inn, tfout, t->outn, tfout);
  if (fin)
    cuaddvec(fout, fin, inn, fin);
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
