#include <stdio.h>
#include <string.h>

#include "cholo.hh"
#include "cudamem.hh"
#include "random.hh"

namespace makemore {

static void cpuchol(double *U, unsigned int dim) {
  unsigned int i, j, k; 

  for (k = 0; k < dim; k++) {
    assert(U[k * dim + k] > 0);
    U[k * dim + k] = sqrt(U[k * dim + k]);
    assert(U[k * dim + k] > 0);
    
    for (j = (k + 1); j < dim; j++)
      U[k * dim + j] /= U[k * dim + k];
             
    for (i = (k + 1); i < dim; i++)
      for (j = i; j < dim; j++)
        U[i * dim + j] -= U[k * dim + i] * U[k * dim + j];

  }

  for (i = 0; i < dim; i++)
    for (j = 0; j < i; j++)
      U[i * dim + j] = 0.0;
}

void matinv(double *m, double *x, unsigned int n) {
  memset(x, 0, n * n * sizeof(double));

  for (unsigned int k = 0; k < n; ++k) {
    x[k * n + k] = 1.0 / m[k * n + k];

    for (unsigned int i = k + 1; i < n; ++i) {
      x[k * n + i] = 0;
      double a = 0, b = 0;
      for (unsigned int j = k; j < i; ++j) {
        double a = -m[j * n + i];
        double b = x[k * n + j];
        double c = m[i * n + i];
        x[k * n + i] += a * b / c;
      }
    }
  }
}

#if 0
1. for k = 1 to n
2.   L[k,k] = 1/L[k,k]
3.   for i = k+1 to n
4.     L[i, k] = -L[i, k:i-1]*L[k:i-1, k]/L[k, k]
5.   end for i
6. end for k
#endif

Cholo::Cholo(unsigned int _dim) : dim(_dim) {
  dim2 = dim * dim;
  tmp = new double[dim];
  cumake(&in, dim);
  cumake(&out, dim);
  cumake(&mean, dim);
  cumake(&var, dim);
  cumake(&cov, dim2);
  cumake(&chol, dim2);
  cumake(&unchol, dim2);
  ichol = 0;

  reset();
}

Cholo::~Cholo() {
  cufree(cov);
  cufree(chol);
  cufree(mean);
  cufree(var);
  cufree(in);
  cufree(out);
}

void Cholo::observe(const double *x) {
  encude(x, dim, in);
  observecu(in);
}

void Cholo::observecu(const double *x) {
  cuupmean(x, 1, dim, ichol, mean);
  cuupvariance(x, 1, dim, ichol, var);
  cuupcovariance(x, 1, dim, ichol, 8, cov);
  ++ichol;

#if 0
  ++ichol;
  if (ichol >= nchol) {
    finalize();
    ichol = 0;
  }
#endif
}

void Cholo::reset() {
  cuzero(cov, dim2);
  cuzero(chol, dim2);

  cuzero(var, dim);
#if 0
  double *one = new double[dim];
  for (unsigned int i = 0; i < dim; ++i)
    one[i] = 1.0;
  encude(one, dim, var);
  delete[] one;
#endif

  cuzero(mean, dim);
}

void Cholo::finalize() {
  double *tmp2;
  double *tmp1;

  cumake(&tmp1, dim);
  cumake(&tmp2, dim2);

  cuzero(tmp2, dim2);

  for (unsigned int row = 0; row < dim; ++row)
    cucopy(mean, dim, tmp2 + row * dim); 
  cumatxpose(tmp2, dim, dim, chol);

  for (unsigned int row = 0; row < dim; ++row)
    cumulvec(chol + row * dim, mean, dim, chol + row * dim); 
  cumuld(chol, -1, dim2, chol);

  cuaddvec(cov, chol, dim2, chol);

  cumulvec(mean, mean, dim, tmp1);
  cusubvec(var, tmp1, dim, var);

  for (unsigned int row = 0; row < dim; ++row)
    cudivsqrtvec(chol + row * dim, var, dim, tmp2 + row * dim); 
  cumatxpose(tmp2, dim, dim, chol);

  for (unsigned int row = 0; row < dim; ++row)
    cudivsqrtvec(chol + row * dim, var, dim, tmp2 + row * dim); 
  cumatxpose(tmp2, dim, dim, chol);
  cucopy(chol, dim2, cov);

  cufree(tmp2);
  cufree(tmp);

#if 1
  tmp2 = new double[dim2];
  decude(chol, dim2, tmp2);
  cpuchol(tmp2, dim);
  encude(tmp2, dim2, chol);
  delete[] tmp2;
#else
  cuchol(chol, dim);
#endif
}

void Cholo::generate(double *x, double m) {
  for (unsigned int i = 0; i < dim; ++i)
    tmp[i] = randgauss() * m;
  generate(tmp, x);
}

void Cholo::generate(const double *y, double *x) {
  encude(y, dim, in);

  cumatxvec(chol, in, dim, dim, out);
  cumulsqrtvec(out, var, dim, out);
  cuaddvec(out, mean, dim, out);
  decude(out, dim, x);
}

void Cholo::encode(const double *x, double *y) {
  encude(x, dim, in);

  cusubvec(in, mean, dim, out);
  cudivsqrtvec(out, var, dim, out);
  cumatxvec(unchol, out, dim, dim, out);
  decude(out, dim, y);
}

void Cholo::save(FILE *fp) {
  size_t ret;

  decude(mean, dim, tmp);
  ret = fwrite(tmp, sizeof(double), dim, fp);
  assert(ret == dim);

  decude(var, dim, tmp);
  ret = fwrite(tmp, sizeof(double), dim, fp);
  assert(ret == dim);

  double *tmp2 = new double[dim2];

  decude(chol, dim2, tmp2);
for (unsigned int i = 0; i < dim2; ++i) {
  assert(!isnan(tmp2[i]));
}
  ret = fwrite(tmp2, sizeof(double), dim2, fp);
  assert(ret == dim2);

  decude(cov, dim2, tmp2);
for (unsigned int i = 0; i < dim2; ++i) {
  assert(!isnan(tmp2[i]));
}
  ret = fwrite(tmp2, sizeof(double), dim2, fp);
  assert(ret == dim2);

  delete[] tmp2;
}

void Cholo::load(FILE *fp) {
  size_t ret;

  ret = fread(tmp, sizeof(double), dim, fp);
  assert(ret == dim);

double *meantmp = new double[dim];
memcpy(meantmp, tmp, sizeof(double) * dim);

	  encude(tmp, dim, mean);

	  ret = fread(tmp, sizeof(double), dim, fp);
	  assert(ret == dim);
	  encude(tmp, dim, var);

for (unsigned int i = 0; i < dim; ++i) {
fprintf(stderr, "cholo mean[%u] = %lf var[%u] = %lf\n", i, meantmp[i], i, tmp[i]);
}

	  double *tmp2 = new double[dim2];

  ret = fread(tmp2, sizeof(double), dim2, fp);
  assert(ret == dim2);
for (unsigned int i = 0; i < dim2; ++i) {
  if (isnan(tmp2[i])) { fprintf(stderr, "isnan(tmp2[%u])\n", i); tmp2[i] = 0; }
}
for (unsigned int i = 0; i < dim2; ++i) {
  assert(!isnan(tmp2[i]));
  if (tmp2[i] < -100.0 || tmp2[i] > 100.0) {
    fprintf(stderr, "tmp2[%u]=%lf\n", i, tmp2[i]);
  }
}
  encude(tmp2, dim2, chol);
double *tmp3 = new double[dim2];
matinv(tmp2, tmp3, dim);
encude(tmp3, dim2, unchol);
delete[] tmp3;

  ret = fread(tmp2, sizeof(double), dim2, fp);
  assert(ret == dim2);
for (unsigned int i = 0; i < dim2; ++i) {
  if (isnan(tmp2[i])) { fprintf(stderr, "isnan(tmp2[%u])\n", i); tmp2[i] = 0; }
}
for (unsigned int i = 0; i < dim2; ++i) {
  assert(!isnan(tmp2[i]));
  if (tmp2[i] < -100.0 || tmp2[i] > 100.0) {
    fprintf(stderr, "tmp2[%u]=%lf\n", i, tmp2[i]);
  }
}
  encude(tmp2, dim2, cov);

  delete[] tmp2;
}

void Cholo::load(const std::string &fn) {
  FILE *fp = fopen(fn.c_str(), "r");
  assert(fp);
  this->load(fp);
  fclose(fp);
}

}
