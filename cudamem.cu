#define __MAKEMORE_CUDAMEM_CU__ 1
#include "cudamem.hh"

#include <assert.h>
#include <stdio.h>

void encudev(const void *a, unsigned int n, void *da) {
  cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
}

void decudev(const void *da, unsigned int n, void *a) {
  cudaMemcpy(a, da, n, cudaMemcpyDeviceToHost);
}

void cumakev(void **dp, unsigned int n) {
  void *d = NULL;
  assert(0 == cudaMalloc((void **)&d, n));
  assert(d != NULL);
  assert(dp != NULL);
  *dp = d;
}

void cufreev(void *x) {
  cudaFree(x);
}

void cuzerov(void *x, unsigned int n) {
  cudaMemset((void *)x, 0, n);
}

void cucopyv(const void *x, unsigned int n, void *y) {
  cudaMemcpy(y, x, n, cudaMemcpyDeviceToDevice);
}


__global__ void gpu_cuaddvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

void cuaddvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cuaddvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_cusubvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] - b[i];
}

void cusubvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cusubvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void gpu_cumulvec(const double *a, double m, int n, double *b) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    b[i] = a[i] * m;
}

void cumulvec(const double *a, double m, unsigned int n, double *b) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cumulvec<<<gs, bs>>>(a, m, n, b);
}


__global__ void gpu_cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  unsigned int ccol = blockIdx.x * blockDim.x + threadIdx.x;
  if (ccol >= ccols)
    return;

  unsigned int k = ccols - bcols;
  if (ccol < k) {
    unsigned int acol = ccol;
    for (unsigned int row = 0; row < rows; ++row) {
      c[ccols * row + ccol] = a[acols * row + acol];
    }
  } else {
    unsigned int bcol = ccol - k;
    for (unsigned int row = 0; row < rows; ++row) {
      c[ccols * row + ccol] = b[bcols * row + bcol];
    }
  }
}

void cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  int bs = 128;
  int gs = ((ccols + bs - 1) / bs);
  gpu_cucutpaste<<<gs, bs>>>(a, b, rows, acols, bcols, ccols, c);
}


__global__ void gpu_cucutadd(
  const double *a, unsigned int rows, unsigned int acols,
  unsigned int bcols, double *b
) {
  unsigned int bcol = blockIdx.x * blockDim.x + threadIdx.x;
  if (bcol >= bcols)
    return;

  unsigned int acol = bcol + (acols - bcols);
  for (unsigned int row = 0; row < rows; ++row) {
    b[bcols * row + bcol] += a[acols * row + acol];
  }
}

void cucutadd(
  const double *a, unsigned int rows, unsigned int acols,
  unsigned int bcols, double *b
) {
  int bs = 128;
  int gs = ((bcols + bs - 1) / bs);
  gpu_cucutadd<<<gs, bs>>>(a, rows, acols, bcols, b);
}


__global__ void gpu_sumsq(
  const double *a, unsigned int n, double *sumsqp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = i0 + 128 >= n ? n : i0 + 128;
  
  double s = 0;
  for (unsigned int i = i0; i < i1; ++i)
    s += a[i] * a[i];
  sumsqp[si] = s;
}

double cusumsq(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *sumsqp;
  unsigned int sumsqn = ((n + 127) / 128);
  cumake(&sumsqp, sumsqn);

  int bs = 128;
  int gs = (sumsqn + bs - 1) / bs;
  gpu_sumsq<<<gs, bs>>>(a, n, sumsqp);

  double *sumsqv = new double[sumsqn];
  decude(sumsqp, sumsqn, sumsqv);

  double s = 0;
  for (int i = 0; i < sumsqn; ++i)
    s += sumsqv[i];

  cufree(sumsqp);
  delete[] sumsqv;

  return s;
}


__global__ void gpu_max(
  const double *a, unsigned int n, double *maxp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = i0 + 128 >= n ? n : i0 + 128;

  unsigned int i = i0;
  double s = fabs(a[i]);
  ++i;
  
  for (; i < i1; ++i) {
    double aa = fabs(a[i]);
    if (aa > s)
      s = aa;
  }

  maxp[si] = s;
}

double cumaxabs(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *maxp;
  unsigned int maxn = ((n + 127) / 128);

  cumake(&maxp, maxn);

  int bs = 128;
  int gs = (maxn + bs - 1) / bs;
  gpu_max<<<gs, bs>>>(a, n, maxp);

  double *maxv = new double[maxn];
  decude(maxp, maxn, maxv);

  double s = maxv[0];
  for (int i = 1; i < maxn; ++i)
    if (maxv[i] > s)
      s = maxv[i];

  cufree(maxp);
  delete[] maxv;

  return s;
}
