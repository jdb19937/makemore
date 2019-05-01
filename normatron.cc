#define __MAKEMORE_NORMATRON_CC__ 1

#include "normatron.hh"
#include "cudamem.hh"

namespace makemore {

Normatron::Normatron(Mapfile *_mapfile, unsigned int _n, unsigned int _mbn, double _beta) : Tron(_n * _mbn, _n * _mbn) {
  n = _n;
  beta = _beta;
  mbn = _mbn;
  mapfile = _mapfile;

  assert(inn == outn);
  assert(n * mbn == inn);

  in = NULL;
  fin = NULL;

  cumake(&out, n * mbn);
  cumake(&fout, n * mbn);

  cumake(&mean, n);
  mapfile->map(mean, n);
  mapfile->load(mean);


  cumake(&var, n);
  mapfile->map(var, n);
  mapfile->load(var);

double *tmp = new double[n];
double *tmp2 = new double[n];
decude(mean, n, tmp);
decude(var, n, tmp2);
for (unsigned int j = 0; j < n; ++j) {
fprintf(stderr, "normatron mean[%u] = %lf    var[%u] = %lf\n", j, tmp[j], j, tmp2[j]);
}
delete[] tmp;
delete[] tmp2;
}

Normatron::~Normatron() {
  cufree(mean);
  cufree(var);

  cufree(out);
  cufree(fout);
}

const double *Normatron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    cusubvec(in + mbi * n, mean, n, out + mbi * n);
    cudivsqrtvec(out + mbi * n, var, n, out + mbi * n);
  }

  cuzero(fout, outn);

  return out;
}

void Normatron::train(double nu) {
  nu *= beta;

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    if (fin)
      cumulsqrtvec(fout + mbi * n, var, n, fin + mbi * n);

    cuupmeanexp(in + mbi * n, n, nu, mean);
    cuupvarexp(in + mbi * n, mean, n, nu, var);
  }
}

void Normatron::randomize(double disp) {
  cuzero(mean, n);
  
  double *one = new double[n];
  for (unsigned int i = 0; i < n; ++i)
    one[i] = 1.0;
  encude(one, n, var);
  delete[] one;
}

}
