#include "cudamem.hh"

#include <assert.h>
#include <stdio.h>

void encude(const double *a, unsigned int n, double *da) {
  cudaMemcpy(da, a, sizeof(double)*n, cudaMemcpyHostToDevice);
}

void decude(const double *da, unsigned int n, double *a) {
  cudaMemcpy(a, da, sizeof(double)*n, cudaMemcpyDeviceToHost);
}

double *cumake(unsigned int n) {
  double *d = NULL;
  assert(0 == cudaMalloc((void **)&d, sizeof(double) * n));
  assert(d != NULL);
  return d;
}

void cufree(double *x) {
  cudaFree((void *)x);
}

void cuzero(double *x, unsigned int n) {
  cudaMemset((void *)x, 0, n * sizeof(double));
}

void cucopy(const double *x, unsigned int n, double *y) {
  cudaMemcpy(y, x, sizeof(double) * n, cudaMemcpyDeviceToDevice);
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
