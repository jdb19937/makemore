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


__global__ void gpu_cuaddvec(const double *a, const double *b, double *c, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
      c[i] = a[i] + b[i];
}

void cuaddvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 128;
  int gs = ((n + bs - 1) / bs);
  gpu_cuaddvec<<<gs, bs>>>(a, b, c, n);
}

__global__ void gpu_cucutpaste(
  const double *a, const double *b,
  unsigned int rows, unsigned int acols, unsigned int bcols, unsigned int ccols,
  double *c
) {
  unsigned int ccol = blockIdx.x * blockDim.x + threadIdx.x;
  if (ccol < ccols)
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
  if (bcol < bcols)
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

