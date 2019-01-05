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
//fprintf(stderr, "makev %u -> ", n);
  assert(0 == cudaMalloc((void **)&d, n));
//fprintf(stderr, "%llu\n", d);
  assert(d != NULL);
  assert(dp != NULL);
  *dp = d;
}

void cufreev(void *x) {
//fprintf(stderr, "cufree %llu\n", x);
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
  unsigned int ci = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int cn = ccols * rows;
  if (ci >= cn)
    return;
  unsigned int ccol = ci % ccols;
  unsigned int row = ci / ccols;

  unsigned int k = ccols - bcols;
  if (ccol < k) {
    unsigned int acol = ccol;
    unsigned int ai = acols * row + acol;
    c[ci] = a[ai];
  } else {
    unsigned int bcol = ccol - k;
    unsigned int bi = bcols * row + bcol;
    c[ci] = b[bi];
  }
}

void cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  assert(ccols >= bcols);
  unsigned int cn = ccols * rows;

  int bs = 128;
  int gs = ((cn + bs - 1) / bs);
  gpu_cucutpaste<<<gs, bs>>>(a, b, rows, acols, bcols, ccols, c);
}


__global__ void gpu_cucutadd(
  const double *a,
  unsigned int rows, unsigned int acols, unsigned int bcols,
  double *b
) {
  unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int bn = bcols * rows;
  if (bi >= bn)
    return;
  unsigned int bcol = bi % bcols;
  unsigned int row = bi / bcols;

  unsigned int acol = bcol + (acols - bcols);
  unsigned int ai = acols * row + acol;

  b[bi] += a[ai];
}

void cucutadd(
  const double *a,
  unsigned int rows, unsigned int acols, unsigned int bcols,
  double *b
) {
  assert(acols >= bcols);

  unsigned int bn = bcols * rows;
  int bs = 128;
  int gs = ((bn + bs - 1) / bs);
  gpu_cucutadd<<<gs, bs>>>(a, rows, acols, bcols, b);
}


__global__ void gpu_sumsq(
  const double *a, unsigned int n, double *sumsqp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);
  
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

  double *sumsqp = NULL;
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


__global__ void gpu_maxabs(
  const double *a, unsigned int n, double *maxp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);

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

  double *maxp = NULL;
  unsigned int maxn = ((n + 127) / 128);
  cumake(&maxp, maxn);

  int bs = 128;
  int gs = (maxn + bs - 1) / bs;
  gpu_maxabs<<<gs, bs>>>(a, n, maxp);

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

__global__ void gpu_meandevcols(
  const double *a, unsigned int rows, unsigned int cols, double *m, double *d
) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  double mean = 0;
  for (unsigned int row = 0; row < rows; ++rows)
    mean += a[row * cols + col]; 
  mean /= (double)rows;
  m[col] = mean;

  double dev = 0;
  for (unsigned int row = 0; row < rows; ++rows) {
    double z = a[row * cols + col] - mean;
    dev += z * z;
  }
  dev /= (double)rows;
  dev = sqrt(dev);
  d[col] = dev;
}

void cumeandevcols(
  const double *a, unsigned int rows, unsigned int cols, double *m, double *d
) {
  assert(rows > 0);
  assert(cols > 0);

  int bs = 128;
  int gs = ((cols + bs - 1) / bs);
  gpu_meandevcols<<<gs, bs>>>(a, rows, cols, m, d);
}


__global__ void gpu_normalize(
  const double *a, unsigned int rows, unsigned int cols,
  const double *m, const double *d, double *b
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols)
    return;
  unsigned int col = i % cols;

  double x = a[i];
  x -= m[col];
  x /= d[col];
  b[i] = x;
}

void cunormalize(
  const double *a, unsigned int rows, unsigned int cols,
  const double *m, const double *d, double *b
) {
  assert(rows > 0);
  assert(cols > 0);
  unsigned int n = rows * cols;

  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_normalize<<<gs, bs>>>(a, rows, cols, m, d, b);
}
