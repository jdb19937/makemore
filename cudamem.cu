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
